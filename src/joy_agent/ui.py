"""Terminal chat input, streaming preview, and Rich output."""

from __future__ import annotations

import asyncio
import difflib
import logging
import queue
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.filters import Always, Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension, ScrollOffsets
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame
from rich import box
from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .llm import StreamHandle, StreamingLLM, _safe_error
from .tools import FileChange

logger = logging.getLogger(__name__)

INPUT_PLACEHOLDER = "Ask Joy to do something…"
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
ADDED_BACKGROUND = "#263d30"
ADDED_FOREGROUND = "#a3c9a8"
DELETED_BACKGROUND = "#452c30"
DELETED_FOREGROUND = "#d9a0a5"
CHANGED_MARKER_BACKGROUND = "#303b4a"
CHANGED_MARKER_FOREGROUND = "#9cb9d8"
NORMAL_BACKGROUND = "#1e1e24"
NORMAL_FOREGROUND = "#c5cbd3"
UNCHANGED_FOREGROUND = "#858b98"
ADDED_COUNT_FOREGROUND = ADDED_FOREGROUND
DELETED_COUNT_FOREGROUND = DELETED_FOREGROUND


@dataclass
class ChatState:
    """Mutable state shared by the prompt layout and stream task."""

    llm: StreamingLLM | None = None
    pending: bool = False
    spinner_index: int = 0
    preview_text: str = ""
    generation_id: int = 0
    stream_handle: StreamHandle | None = None
    reply_task: asyncio.Task[None] | None = None
    pending_user_text: str = ""
    tool_statuses: list[tuple[str, str]] = None
    tool_thinking: str = ""
    tool_started_at: float | None = None
    tool_elapsed: float | None = None
    tool_finished_at: datetime | None = None
    tool_failed: bool = False
    last_tool_action: str = ""
    last_tool_target: str = ""
    tool_used: bool = False
    interruption_recorded: bool = False

    def __post_init__(self) -> None:
        if self.tool_statuses is None:
            self.tool_statuses = []


def _input_key_bindings(state: ChatState) -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("enter", eager=True)
    def submit(event) -> None:
        _submit_buffer(event, state)

    @bindings.add("escape", "enter", eager=True)
    def insert_newline(event) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c", eager=True)
    def cancel_input(event) -> None:
        if state.pending:
            _cancel_pending(event, state)
        else:
            event.current_buffer.reset()

    return bindings


def _submit_buffer(event, state: ChatState) -> None:
    """Start a stream, leaving the buffer untouched while one is pending."""

    if state.pending or state.llm is None:
        return

    buffer = event.current_buffer
    text = buffer.text
    if not text.strip():
        return
    buffer.reset()

    if text.strip() in {"/quit", "/exit"}:
        event.app.exit()
        return

    state.generation_id += 1
    state.pending = True
    state.preview_text = ""
    state.pending_user_text = text
    state.tool_statuses = []
    state.tool_thinking = ""
    state.tool_started_at = None
    state.tool_elapsed = None
    state.tool_finished_at = None
    state.tool_failed = False
    state.last_tool_action = ""
    state.last_tool_target = ""
    state.tool_used = False
    state.interruption_recorded = False
    state.stream_handle = state.llm.start_stream(
        state.llm.messages_for(text), state.generation_id
    )
    state.reply_task = event.app.create_background_task(
        _stream_submission(event.app, state, text, event.app.joy_console)
    )
    event.app.invalidate()


def _cancel_pending(event, state: ChatState) -> None:
    """Cancel UI delivery immediately and preserve the current draft."""

    partial = state.preview_text
    if state.stream_handle is not None:
        state.stream_handle.cancel()
    commit_interruption = getattr(state.llm, "commit_interruption", None)
    if (
        state.tool_used
        and not state.interruption_recorded
        and state.pending_user_text
        and callable(commit_interruption)
    ):
        commit_interruption(state.pending_user_text)
        state.interruption_recorded = True
    state.generation_id += 1
    state.pending = False
    state.preview_text = ""
    task = state.reply_task
    state.reply_task = None
    if task is not None:
        task.cancel()
    event.app.invalidate()
    event.app.create_background_task(
        _render_terminal_notice(
            event.app,
            event.app.joy_console,
            partial,
            "Response cancelled.",
        )
    )


def _style() -> Style:
    return Style.from_dict(
        {
            "frame.border": "fg:#393939 bg:#393939",
            "input": "fg:#ffffff bg:#393939",
            "placeholder": "fg:#999999 bg:#393939",
            "prompt": "fg:#ffffff bg:#393939",
            "thinking": f"fg:{NORMAL_FOREGROUND}",
            "preview": f"fg:{NORMAL_FOREGROUND}",
            "tool": f"fg:{UNCHANGED_FOREGROUND}",
        }
    )


def _indented_lines(text: str) -> str:
    return "\n".join(f" {line}" for line in text.split("\n"))


def _normalize_inline_bullets(text: str) -> str:
    """Turn obvious inline capability lists into Markdown list items."""

    lines: list[str] = []
    list_open = False
    for line in text.split("\n"):
        if list_open and line.strip():
            if lines and lines[-1] != "":
                lines.append("")
            list_open = False
        if line.count(" - ") < 2 or not re.match(r"^\s*[^-\n]+:\s+-\s+", line):
            lines.append(line)
            continue
        prefix, items_text = line.split(" - ", 1)
        items = re.split(r"\s+-\s+", items_text)
        lines.extend([prefix.rstrip(), "", *(f"- {item.strip()}" for item in items)])
        list_open = True
    if list_open and lines and lines[-1].startswith("-"):
        lines.append("")
    return "\n".join(lines)


def _tool_status_text(state: ChatState) -> str:
    if not state.tool_statuses:
        return ""
    name, status = state.tool_statuses[-1]
    if status == "running":
        seconds = 0
        if state.tool_started_at is not None:
            seconds = max(0, int(time.monotonic() - state.tool_started_at))
        return f"  Working ({seconds}s)"
    if status in {"finished", "failed"} and state.tool_elapsed is not None:
        seconds = max(1, int(round(state.tool_elapsed)))
        finished_at = state.tool_finished_at or datetime.now()
        label = "Worked" if status == "finished" else "Failed"
        return f"  {label} for {seconds}s · {finished_at:%H:%M}"
    return f"  {name} · {status}"


def create_prompt_session(state: ChatState | None = None) -> PromptSession[str]:
    """Create the existing fixed-height input plus streaming preview."""

    chat_state = state if state is not None else ChatState()
    session = PromptSession(
        message=[("class:prompt", "› ")],
        multiline=True,
        placeholder=[("class:placeholder", INPUT_PLACEHOLDER)],
        key_bindings=_input_key_bindings(chat_state),
        style=_style(),
        color_depth=ColorDepth.TRUE_COLOR,
    )

    for window in _default_buffer_windows(session):
        window.height = Dimension.exact(1)
        window.dont_extend_height = Always()
        window.style = "class:input"

    input_frame = Frame(
        session.layout.container,
        height=Dimension.exact(3),
    ).container
    thinking = ConditionalContainer(
        Window(
            FormattedTextControl(
                lambda: [
                    (
                        "class:thinking",
                        f"{SPINNER_FRAMES[chat_state.spinner_index]} "
                        f"{chat_state.tool_thinking or 'Thinking…'}",
                    )
                ]
            ),
            height=1,
        ),
        filter=Condition(
            lambda: (
                chat_state.pending
                and not chat_state.preview_text
                and not chat_state.tool_used
            )
        ),
    )
    preview = ConditionalContainer(
        Window(
            FormattedTextControl(
                lambda: [("class:preview", _indented_lines(chat_state.preview_text))]
            ),
            height=Dimension(min=1, max=10),
            scroll_offsets=ScrollOffsets(bottom=0),
            wrap_lines=True,
        ),
        filter=Condition(lambda: chat_state.pending and bool(chat_state.preview_text)),
    )
    tool_status = ConditionalContainer(
        Window(
            FormattedTextControl(
                lambda: [
                    (
                        "class:tool",
                        _tool_status_text(chat_state),
                    )
                ]
            ),
            height=Dimension.exact(1),
            dont_extend_height=Always(),
        ),
        filter=Condition(lambda: chat_state.pending and bool(chat_state.tool_statuses)),
    )
    loading_gap = ConditionalContainer(
        Window(FormattedTextControl(""), height=1),
        filter=Condition(lambda: chat_state.pending),
    )
    session.layout.container = HSplit(
        [thinking, preview, tool_status, loading_gap, input_frame]
    )
    session.joy_state = chat_state
    session.joy_input_frame = input_frame
    return session


def _default_buffer_windows(session: PromptSession[str]) -> Iterable:
    return (
        window
        for window in session.layout.find_all_windows()
        if isinstance(window.content, BufferControl)
        and window.content.buffer is session.default_buffer
    )


def render_user_message(text: str, console: Console | None = None) -> None:
    output = console if console is not None else Console()
    output.print(
        Panel(
            Text(f"› {text}", style="white"),
            box=box.SQUARE,
            border_style="#393939",
            padding=(0, 1),
            style="white on #393939",
            expand=True,
        )
    )
    output.print()


def render_markdown(text: str, console: Console | None = None) -> None:
    output = console if console is not None else Console()
    formatted = _normalize_inline_bullets(text)
    output.print(Markdown(formatted))


def render_syntax(
    code: str,
    lexer: str = "text",
    console: Console | None = None,
    line_numbers: bool = True,
) -> None:
    """Render source code with Rich syntax highlighting."""
    output = console if console is not None else Console()
    output.print(
        Syntax(
            code,
            lexer,
            line_numbers=line_numbers,
            theme="ansi_dark",
            word_wrap=False,
        )
    )
    output.print()


def render_code_preview(
    code: str,
    path: str = "",
    console: Console | None = None,
) -> None:
    """Render a code preview using a file-extension lexer when available."""
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    lexer = {
        "bash": "bash",
        "css": "css",
        "html": "html",
        "js": "javascript",
        "json": "json",
        "md": "markdown",
        "py": "python",
        "rb": "ruby",
        "rs": "rust",
        "sh": "bash",
        "sql": "sql",
        "toml": "toml",
        "ts": "typescript",
        "yaml": "yaml",
        "yml": "yaml",
    }.get(suffix, "text")
    render_syntax(code, lexer, console)


def render_pretty(value: Any, console: Console | None = None) -> None:
    """Render a structured value using Rich's readable pretty printer."""
    output = console if console is not None else Console()
    output.print(Pretty(value, indent_guides=True))
    output.print()


def render_table(
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    title: str | None = None,
    console: Console | None = None,
) -> None:
    """Render structured rows as a consistently styled Rich table."""
    if not columns:
        raise ValueError("table requires at least one column")
    output = console if console is not None else Console()
    table = Table(
        title=title,
        box=box.SIMPLE,
        border_style=UNCHANGED_FOREGROUND,
        header_style=f"bold {NORMAL_FOREGROUND}",
        show_lines=False,
    )
    for column in columns:
        table.add_column(str(column))
    for row in rows:
        if len(row) != len(columns):
            raise ValueError("table row width must match the column count")
        table.add_row(*(Text(str(value)) for value in row))
    output.print(table)
    output.print()


def render_assistant_message(text: str, console: Console | None = None) -> None:
    output = console if console is not None else Console()
    render_markdown(text, output)
    output.print()


def _styled_change_line(
    marker: str,
    line_number: int,
    line_number_width: int,
    line: str,
    background: str,
    foreground: str,
    width: int,
) -> Text:
    rendered = Text()
    line_prefix = f"{line_number:>{line_number_width}} "
    rendered.append(
        line_prefix,
        style=f"{UNCHANGED_FOREGROUND} on {background}",
    )
    rendered.append(
        marker,
        style=f"{foreground} on {background}",
    )
    rendered.append(" ", style=f"{foreground} on {background}")
    rendered.append(line, style=f"{foreground} on {background}")
    padding = max(0, width - cell_len(line_prefix + marker + " " + line))
    rendered.append(" " * padding, style=f"on {background}")
    return rendered


def _modified_change_line(
    old_line: str,
    new_line: str,
    line_number: int,
    line_number_width: int,
    width: int,
) -> Text:
    rendered = Text()
    line_prefix = f"{line_number:>{line_number_width}} "
    rendered.append(
        line_prefix,
        style=f"{UNCHANGED_FOREGROUND} on {ADDED_BACKGROUND}",
    )
    rendered.append(
        "+",
        style=f"{ADDED_FOREGROUND} on {ADDED_BACKGROUND}",
    )
    rendered.append(" ", style=f"{ADDED_FOREGROUND} on {ADDED_BACKGROUND}")
    matcher = difflib.SequenceMatcher(None, old_line, new_line, autojunk=False)
    for tag, _, _, new_start, new_end in matcher.get_opcodes():
        if tag == "delete":
            continue
        style = (
            f"{CHANGED_MARKER_FOREGROUND} on {ADDED_BACKGROUND}"
            if tag != "equal"
            else f"{ADDED_FOREGROUND} on {ADDED_BACKGROUND}"
        )
        rendered.append(new_line[new_start:new_end], style=style)
    padding = max(
        0,
        width - cell_len(line_prefix + "+ " + new_line),
    )
    rendered.append(" " * padding, style=f"on {ADDED_BACKGROUND}")
    return rendered


def _change_stats(change: FileChange) -> tuple[int, int]:
    before_lines = [] if change.before is None else change.before.splitlines()
    after_lines = change.after.splitlines()
    matcher = difflib.SequenceMatcher(
        None, before_lines, after_lines, autojunk=False
    )
    added = 0
    deleted = 0
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "delete":
            deleted += old_end - old_start
        elif tag == "insert":
            added += new_end - new_start
        elif tag == "replace":
            deleted += old_end - old_start
            added += new_end - new_start
    return added, deleted


def _change_action(change: FileChange, added: int, deleted: int) -> str:
    if change.before is None:
        return "Created"
    if added and not deleted:
        return "Added"
    if deleted and not added:
        return "Deleted"
    return "Edited"


def _activity_line(
    action: str,
    target: str = "",
    prefix: str = "",
) -> Text:
    rendered = Text("•", style=f"{UNCHANGED_FOREGROUND} on {NORMAL_BACKGROUND}")
    rendered.append(
        f" {prefix}{action}",
        style=f"{NORMAL_FOREGROUND} on {NORMAL_BACKGROUND}",
    )
    if target:
        rendered.append(" ", style=f"{NORMAL_FOREGROUND} on {NORMAL_BACKGROUND}")
        rendered.append(target, style=f"{NORMAL_FOREGROUND} on {NORMAL_BACKGROUND}")
    return rendered


def _change_header(change: FileChange, added: int, deleted: int) -> Text:
    action = _change_action(change, added, deleted)
    rendered = _activity_line(action, change.path)
    rendered.append(" (", style=f"{NORMAL_FOREGROUND} on {NORMAL_BACKGROUND}")
    rendered.append(
        f"+{added}",
        style=f"{ADDED_COUNT_FOREGROUND} on {NORMAL_BACKGROUND}",
    )
    rendered.append(", ", style=f"{NORMAL_FOREGROUND} on {NORMAL_BACKGROUND}")
    rendered.append(
        f"-{deleted}",
        style=f"{DELETED_COUNT_FOREGROUND} on {NORMAL_BACKGROUND}",
    )
    rendered.append(")", style=f"{NORMAL_FOREGROUND} on {NORMAL_BACKGROUND}")
    return rendered


def _completion_line(action: str, target: str, success: bool = True) -> Text:
    status = "Done" if success else "Failed"
    verb = action.lower() if action else "tool call"
    rendered = Text("•", style=f"{UNCHANGED_FOREGROUND} on {NORMAL_BACKGROUND}")
    rendered.append(
        f" {status} — {verb}",
        style=f"{NORMAL_FOREGROUND} on {NORMAL_BACKGROUND}",
    )
    if target:
        rendered.append(" ", style=f"{NORMAL_FOREGROUND} on {NORMAL_BACKGROUND}")
        rendered.append(target, style=f"{NORMAL_FOREGROUND} on {NORMAL_BACKGROUND}")
    return rendered


def render_persistent_summary(
    text: str,
    console: Console | None = None,
    leading_blank: bool = False,
) -> None:
    """Render a model summary as one or more persistent bullet lines."""
    output = console if console is not None else Console()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return
    if leading_blank:
        output.print()
    for line in lines:
        rendered = Text("•", style=f"{UNCHANGED_FOREGROUND} on {NORMAL_BACKGROUND}")
        rendered.append(
            f" {line}",
            style=f"{NORMAL_FOREGROUND} on {NORMAL_BACKGROUND}",
        )
        output.print(rendered)
    output.print()


def _tool_intent_summary(action: str, target: str) -> str:
    verb = {
        "Writing": "write",
        "Editing": "edit",
        "Reading": "read",
        "Running": "run",
    }.get(action, action.lower() if action else "continue")
    if target:
        return f"I’ll {verb} {target}."
    return f"I’ll {verb} the requested operation."


def _tool_result_summary(action: str, target: str) -> str:
    if target:
        return f"{action or 'Completed'} {target}."
    return f"{action or 'Completed'} the requested operation."


def _render_change_lines(change: FileChange, width: int) -> list[Text]:
    before_lines = [] if change.before is None else change.before.splitlines()
    after_lines = change.after.splitlines()
    line_number_width = max(
        1,
        len(str(max(len(before_lines), len(after_lines), 1))),
    )
    matcher = difflib.SequenceMatcher(
        None, before_lines, after_lines, autojunk=False
    )
    rendered: list[Text] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            rendered.extend(
                _styled_change_line(
                    "-",
                    old_start + index + 1,
                    line_number_width,
                    line,
                    DELETED_BACKGROUND,
                    DELETED_FOREGROUND,
                    width,
                )
                for index, line in enumerate(before_lines[old_start:old_end])
            )
            continue
        if tag == "insert":
            rendered.extend(
                _styled_change_line(
                    "+",
                    new_start + index + 1,
                    line_number_width,
                    line,
                    ADDED_BACKGROUND,
                    ADDED_FOREGROUND,
                    width,
                )
                for index, line in enumerate(after_lines[new_start:new_end])
            )
            continue

        old_changed = before_lines[old_start:old_end]
        new_changed = after_lines[new_start:new_end]
        paired = min(len(old_changed), len(new_changed))
        for index in range(paired):
            rendered.append(
                _styled_change_line(
                    "-",
                    old_start + index + 1,
                    line_number_width,
                    old_changed[index],
                    DELETED_BACKGROUND,
                    DELETED_FOREGROUND,
                    width,
                )
            )
            rendered.append(
                _modified_change_line(
                    old_changed[index],
                    new_changed[index],
                    new_start + index + 1,
                    line_number_width,
                    width,
                )
            )
        rendered.extend(
            _styled_change_line(
                "-",
                old_start + paired + index + 1,
                line_number_width,
                line,
                DELETED_BACKGROUND,
                DELETED_FOREGROUND,
                width,
            )
            for index, line in enumerate(old_changed[paired:])
        )
        rendered.extend(
            _styled_change_line(
                "+",
                new_start + paired + index + 1,
                line_number_width,
                line,
                ADDED_BACKGROUND,
                ADDED_FOREGROUND,
                width,
            )
            for index, line in enumerate(new_changed[paired:])
        )
    return rendered


def render_file_changes(
    changes: Iterable[FileChange], console: Console | None = None
) -> None:
    """Render file mutation headers, diffs, and completion summaries."""
    output = console if console is not None else Console()
    change_list = list(changes)
    for index, change in enumerate(change_list):
        added, deleted = _change_stats(change)
        output.print(_change_header(change, added, deleted))
        for line in _render_change_lines(change, output.width):
            output.print(line)
        if index < len(change_list) - 1:
            output.print()


def render_tool_completion(
    action: str,
    target: str = "",
    success: bool = True,
    console: Console | None = None,
) -> None:
    """Render a completion or failure summary for a tool call."""
    output = console if console is not None else Console()
    output.print(_completion_line(action, target, success))
    output.print()


def render_tool_elapsed(
    elapsed: float,
    finished_at: datetime,
    success: bool = True,
    console: Console | None = None,
) -> None:
    """Render persistent elapsed time below a tool completion summary."""
    output = console if console is not None else Console()
    seconds = max(1, int(round(elapsed)))
    label = "Worked" if success else "Failed"
    rendered = Text(
        f"  {label} for {seconds}s · {finished_at:%H:%M}",
        style=f"{UNCHANGED_FOREGROUND} on {NORMAL_BACKGROUND}",
    )
    output.print(rendered)
    output.print()


def render_stream_notice(
    partial: str,
    notice: str,
    console: Console | None = None,
) -> None:
    output = console if console is not None else Console()
    content = partial + (f"\n{notice}" if partial else notice)
    output.print(Text(_indented_lines(content)))
    output.print()


async def _render_terminal_notice(app, console, partial: str, notice: str) -> None:
    await run_in_terminal(lambda: render_stream_notice(partial, notice, console))


async def _animate(app, state: ChatState) -> None:
    while state.pending and not state.preview_text:
        await asyncio.sleep(0.1)
        state.spinner_index = (state.spinner_index + 1) % len(SPINNER_FRAMES)
        app.invalidate()


async def _stream_submission(app, state, user_text: str, console: Console) -> None:
    handle = state.stream_handle
    if handle is None or state.llm is None:
        return
    animation = asyncio.create_task(_animate(app, state))
    try:
        await run_in_terminal(lambda: render_user_message(user_text, console))
        while True:
            if handle.generation_id != state.generation_id:
                return
            try:
                event = handle.events.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.03)
                continue
            if event.generation_id != state.generation_id:
                continue
            if event.kind in {"text", "refusal"}:
                state.preview_text += event.content
                app.invalidate()
            elif event.kind == "tool_started":
                state.tool_used = True
                state.tool_statuses = [(event.tool_name, "running")]
                action = getattr(event, "tool_action", "") or (
                    event.tool_name.capitalize()
                )
                target = getattr(event, "tool_target", "")
                summary = state.preview_text.strip() or _tool_intent_summary(
                    action, target
                )
                await run_in_terminal(
                    lambda: render_persistent_summary(summary, console)
                )
                state.preview_text = ""
                state.tool_thinking = ""
                state.tool_started_at = time.monotonic()
                state.tool_elapsed = None
                state.tool_finished_at = None
                state.tool_failed = False
                state.last_tool_action = action
                state.last_tool_target = target
                app.invalidate()
            elif event.kind in {"tool_finished", "tool_failed"}:
                status = "failed" if event.kind == "tool_failed" else "finished"
                if state.tool_statuses:
                    name, _ = state.tool_statuses[-1]
                    if name == event.tool_name:
                        state.tool_statuses = [(name, status)]
                if state.tool_started_at is not None:
                    state.tool_elapsed = max(
                        0, time.monotonic() - state.tool_started_at
                    )
                state.tool_finished_at = datetime.now()
                state.tool_thinking = ""
                state.tool_failed = event.kind == "tool_failed"
                state.tool_statuses = []
                if event.kind == "tool_finished":
                    changes = getattr(event, "changes", [])
                    if changes:
                        await run_in_terminal(
                            lambda: render_file_changes(changes, console)
                        )
                else:
                    await run_in_terminal(
                        lambda: render_tool_completion(
                            getattr(event, "tool_action", event.tool_name),
                            getattr(event, "tool_target", ""),
                            success=False,
                            console=console,
                        )
                    )
                app.invalidate()
            elif event.kind == "complete":
                response = state.preview_text
                history_items = getattr(event, "history_items", [])
                try:
                    state.llm.commit_exchange(user_text, response, history_items)
                except TypeError:
                    state.llm.commit_exchange(user_text, response)
                if state.tool_used:
                    summary = response or _tool_result_summary(
                        state.last_tool_action,
                        state.last_tool_target,
                    )
                    await run_in_terminal(
                        lambda: render_persistent_summary(
                            summary,
                            console,
                            leading_blank=True,
                        )
                    )
                    if (
                        state.tool_elapsed is not None
                        and state.tool_finished_at is not None
                    ):
                        await run_in_terminal(
                            lambda: render_tool_elapsed(
                                state.tool_elapsed,
                                state.tool_finished_at,
                                success=not state.tool_failed,
                                console=console,
                            )
                        )
                else:
                    await run_in_terminal(
                        lambda: render_assistant_message(response, console)
                    )
                return
            elif event.kind == "error":
                if (
                    getattr(event, "tool_used", False)
                    and not state.interruption_recorded
                ):
                    state.llm.commit_interruption(user_text)
                    state.interruption_recorded = True
                logger.error(
                    "Joy request error: %s",
                    _safe_error(event.error or RuntimeError("unknown provider error")),
                )
                await _render_terminal_notice(
                    app, console, state.preview_text, "Request failed. Try again."
                )
                return
            elif event.kind == "incomplete":
                if (
                    getattr(event, "tool_used", False)
                    and not state.interruption_recorded
                ):
                    state.llm.commit_interruption(user_text)
                    state.interruption_recorded = True
                await _render_terminal_notice(
                    app,
                    console,
                    state.preview_text,
                    "Response ended before completion.",
                )
                return
    except asyncio.CancelledError:
        handle.cancel()
        raise
    finally:
        animation.cancel()
        if state.stream_handle is handle:
            state.stream_handle = None
        state.preview_text = ""
        state.tool_thinking = ""
        state.tool_started_at = None
        state.tool_elapsed = None
        state.tool_finished_at = None
        state.tool_failed = False
        state.last_tool_action = ""
        state.last_tool_target = ""
        state.pending = False
        state.pending_user_text = ""
        state.tool_statuses = []
        state.tool_used = False
        state.interruption_recorded = False
        state.reply_task = None
        app.invalidate()


def run_repl(
    llm: StreamingLLM,
    session: PromptSession[str] | None = None,
    console: Console | None = None,
) -> None:
    """Run the terminal chat loop and close the LLM client on exit."""

    output = console if console is not None else Console()
    state = getattr(session, "joy_state", ChatState()) if session else ChatState()
    state.llm = llm
    prompt_session = session if session is not None else create_prompt_session(state)
    prompt_session.app.joy_console = output
    try:
        prompt_session.app.run()
    except (EOFError, KeyboardInterrupt):
        return
    finally:
        llm.close()
