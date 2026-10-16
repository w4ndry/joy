"""Safe workspace tools exposed to Joy."""

from __future__ import annotations

import contextlib
import contextvars
import difflib
import os
import signal
import subprocess
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import aisuite as ai

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_READ_LINES = 2_000
MAX_OUTPUT_CHARS = 20_000
OUTPUT_TRUNCATION_MARKER = "\n[tool output truncated at 20,000 characters]"

_workspace_root = Path.cwd().resolve()
_mutation_lock = threading.RLock()


@dataclass
class _ExecutionContext:
    cancel_event: threading.Event
    active_process: subprocess.Popen[str] | None = None
    changes: list[FileChange] = field(default_factory=list)


_execution_context: contextvars.ContextVar[_ExecutionContext | None] = (
    contextvars.ContextVar("joy_tool_execution_context", default=None)
)


class ToolError(ValueError):
    """An expected, model-visible tool failure."""


@dataclass(frozen=True)
class FileChange:
    """Text content changed by a successful file mutation."""

    path: str
    before: str | None
    after: str


def _bounded(value: str) -> str:
    if len(value) <= MAX_OUTPUT_CHARS:
        return value
    return value[: MAX_OUTPUT_CHARS - len(OUTPUT_TRUNCATION_MARKER)] + (
        OUTPUT_TRUNCATION_MARKER
    )


def _resolve_path(path: str) -> Path:
    if not path.strip():
        raise ToolError("path must not be empty")
    candidate = Path(path)
    base = candidate if candidate.is_absolute() else _workspace_root / candidate
    resolved = base.resolve(strict=False)
    try:
        resolved.relative_to(_workspace_root)
    except ValueError as error:
        raise ToolError("path escapes Joy's launch directory") from error
    return resolved


def _read_bytes(path: Path) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ToolError(f"cannot access {path.name}: {error.strerror}") from error
    if size > MAX_FILE_BYTES:
        raise ToolError("file exceeds the 2 MB tool limit")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ToolError(f"cannot read file: {error.strerror}") from error
    if len(data) > MAX_FILE_BYTES:
        raise ToolError("file exceeds the 2 MB tool limit")
    return data


def _decode(data: bytes) -> tuple[bool, str]:
    bom = data.startswith(b"\xef\xbb\xbf")
    try:
        return bom, data[3:].decode("utf-8") if bom else data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ToolError(f"file is not valid UTF-8: {error.reason}") from error


def _write_atomic(path: Path, data: bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record_change(path: Path, before: str | None, after: str) -> None:
    context = _execution_context.get()
    if context is None or before == after:
        return
    relative_path = path.relative_to(_workspace_root).as_posix()
    context.changes.append(FileChange(relative_path, before, after))


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_fuzzy(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "\n".join(
        line.rstrip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    )
    text = text.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201a": "'",
                "\u201b": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u201e": '"',
                "\u201f": '"',
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2015": "-",
                "\u2212": "-",
                "\u00a0": " ",
                "\u2002": " ",
                "\u2003": " ",
                "\u2004": " ",
                "\u2005": " ",
                "\u2006": " ",
                "\u2007": " ",
                "\u2008": " ",
                "\u2009": " ",
                "\u200a": " ",
                "\u202f": " ",
                "\u205f": " ",
                "\u3000": " ",
            }
        )
    )
    return text


def _line_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for line in text.splitlines(keepends=True):
        spans.append((start, start + len(line)))
        start += len(line)
    if not spans or start < len(text):
        spans.append((start, len(text)))
    return spans


def _fuzzy_replace(content: str, old_text: str, new_text: str) -> str:
    normalized_content = _normalize_line_endings(content)
    normalized_old = _normalize_line_endings(old_text)
    normalized_new = _normalize_line_endings(new_text)
    if not normalized_old.strip():
        raise ToolError("old_text must not be empty or whitespace-only")

    exact_count = normalized_content.count(normalized_old)
    if exact_count:
        if exact_count > 1:
            raise ToolError("old_text is ambiguous; provide more context")
        result = normalized_content.replace(normalized_old, normalized_new, 1)
        if result == normalized_content:
            raise ToolError("replacement makes no change")
        return result

    fuzzy_content = _normalize_fuzzy(normalized_content)
    fuzzy_old = _normalize_fuzzy(normalized_old)
    if not fuzzy_old.strip():
        raise ToolError("old_text must not be empty or whitespace-only")
    occurrences = fuzzy_content.count(fuzzy_old)
    if occurrences == 0:
        raise ToolError("could not find old_text in the file")
    if occurrences > 1:
        raise ToolError("old_text is ambiguous after Unicode normalization")

    match_index = fuzzy_content.index(fuzzy_old)
    replaced = fuzzy_content.replace(fuzzy_old, normalized_new, 1)
    if replaced == normalized_content:
        raise ToolError("replacement makes no change")

    original_lines = normalized_content.splitlines(keepends=True)
    spans = _line_spans(fuzzy_content)
    end_index = match_index + len(fuzzy_old)
    start_line = next(i for i, (_, end) in enumerate(spans) if match_index < end)
    end_line = start_line
    while end_line < len(spans) and spans[end_line][1] < end_index:
        end_line += 1
    segment_start = spans[start_line][0]
    segment_end = spans[end_line][1]
    segment = fuzzy_content[segment_start:segment_end]
    local_index = match_index - segment_start
    changed_segment = (
        segment[:local_index]
        + normalized_new
        + segment[local_index + len(fuzzy_old) :]
    )
    return "".join(original_lines[:start_line]) + changed_segment + "".join(
        original_lines[end_line + 1 :]
    )


def _diff(path: Path, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )


def read(path: str, offset: int = 1, limit: int = 200) -> str:
    """Read a UTF-8 file with line numbers.

    Args:
        path: Workspace-relative file path.
        offset: One-based first line to return.
        limit: Number of lines to return, up to 2,000.
    """
    if offset < 1:
        raise ToolError("offset must be at least 1")
    if limit < 1 or limit > MAX_READ_LINES:
        raise ToolError("limit must be between 1 and 2,000")
    resolved = _resolve_path(path)
    _, content = _decode(_read_bytes(resolved))
    lines = content.splitlines()
    selected = lines[offset - 1 : offset - 1 + limit]
    numbered = "\n".join(
        f"{index}: {line}" for index, line in enumerate(selected, offset)
    )
    return _bounded(numbered)


def write(path: str, content: str) -> str:
    """Create or overwrite a UTF-8 workspace file."""
    resolved = _resolve_path(path)
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_FILE_BYTES:
        raise ToolError("content exceeds the 2 MB tool limit")
    with _mutation_lock:
        if resolved.is_symlink():
            raise ToolError("refusing to overwrite a symlink")
        context = _execution_context.get()
        before: str | None = None
        can_record_change = context is not None and not resolved.exists()
        if context is not None and resolved.exists():
            try:
                _, before = _decode(_read_bytes(resolved))
                can_record_change = True
            except ToolError:
                can_record_change = False
        mode = resolved.stat().st_mode & 0o7777 if resolved.exists() else None
        _write_atomic(resolved, encoded, mode)
        if can_record_change:
            _record_change(resolved, before, content)
    return _bounded(f"wrote {len(encoded)} bytes to {path}")


def edit(path: str, old_text: str, new_text: str) -> str:
    """Replace one unique text occurrence and return a unified diff."""
    resolved = _resolve_path(path)
    with _mutation_lock:
        if resolved.is_symlink():
            raise ToolError("refusing to edit a symlink")
        before_bytes = _read_bytes(resolved)
        bom, before = _decode(before_bytes)
        after_lf = _fuzzy_replace(before, old_text, new_text)
        ending = "\r\n" if "\r\n" in before else "\n"
        after = after_lf.replace("\n", ending) if ending == "\r\n" else after_lf
        if after == before:
            raise ToolError("replacement makes no change")
        if resolved.read_bytes() != before_bytes:
            raise ToolError(
                "file changed while preparing the edit; retry after rereading"
            )
        encoded = ("\ufeff" if bom else "").encode("utf-8") + after.encode("utf-8")
        if len(encoded) > MAX_FILE_BYTES:
            raise ToolError("result exceeds the 2 MB tool limit")
        mode = resolved.stat().st_mode & 0o7777
        _write_atomic(resolved, encoded, mode)
        _record_change(resolved, before, after)
    return _bounded(_diff(resolved, before, after))


def _terminate_process(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1)


def bash(command: str, timeout: int = 30) -> str:
    """Run a noninteractive Bash command with the user's permissions.

    Bash is not sandboxed by Joy's file-path restrictions.

    Args:
        command: Command passed to ``bash -lc``.
        timeout: Timeout in seconds, from 1 through 300.
    """
    if not command.strip():
        raise ToolError("command must not be empty")
    if timeout < 1 or timeout > 300:
        raise ToolError("timeout must be between 1 and 300 seconds")
    context = _execution_context.get()
    process = subprocess.Popen(
        ["/bin/bash", "-lc", command],
        cwd=_workspace_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    if context is not None:
        context.active_process = process
    started = time.monotonic()
    try:
        while True:
            if context is not None and context.cancel_event.is_set():
                _terminate_process(process)
                output = process.stdout.read() if process.stdout else ""
                return _bounded("[cancelled]\n" + output)
            if time.monotonic() - started >= timeout:
                _terminate_process(process)
                output = process.stdout.read() if process.stdout else ""
                return _bounded(f"{output}\n[timeout after {timeout}s]\nexit_code=-1")
            try:
                output, _ = process.communicate(timeout=0.1)
                return _bounded(f"{output}exit_code={process.returncode}")
            except subprocess.TimeoutExpired:
                continue
    finally:
        if context is not None:
            context.active_process = None


@contextlib.contextmanager
def execution_context(cancel_event: threading.Event) -> Iterator[_ExecutionContext]:
    """Expose cancellation to tools while AISuite executes them."""
    context = _ExecutionContext(cancel_event)
    token = _execution_context.set(context)
    try:
        yield context
    finally:
        _execution_context.reset(token)


def _responses_tools() -> list[dict[str, object]]:
    converted = []
    for tool in ai.Tools([read, write, edit, bash]).tools():
        function = tool["function"]
        converted.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function["parameters"],
                "strict": False,
            }
        )
    return converted


TOOL_REGISTRY = ai.Tools([read, write, edit, bash])
TOOL_DEFINITIONS = _responses_tools()

