"""Terminal chat input, streaming preview, and Rich output."""

from __future__ import annotations

import asyncio
import logging
import queue
from collections.abc import Iterable
from dataclasses import dataclass

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
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .llm import StreamHandle, StreamingLLM, _safe_error

logger = logging.getLogger(__name__)

INPUT_PLACEHOLDER = "Ask Joy to do something…"
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


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
            "thinking": "fg:#aaaaaa",
            "preview": "fg:#ffffff",
        }
    )


def _indented_lines(text: str) -> str:
    return "\n".join(f" {line}" for line in text.split("\n"))


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
                        f"{SPINNER_FRAMES[chat_state.spinner_index]} Thinking…",
                    )
                ]
            ),
            height=1,
        ),
        filter=Condition(lambda: chat_state.pending and not chat_state.preview_text),
    )
    preview = ConditionalContainer(
        Window(
            FormattedTextControl(
                lambda: [("class:preview", _indented_lines(chat_state.preview_text))]
            ),
            height=Dimension(min=1, max=10),
            scroll_offsets=ScrollOffsets(bottom=0),
        ),
        filter=Condition(lambda: chat_state.pending and bool(chat_state.preview_text)),
    )
    loading_gap = ConditionalContainer(
        Window(FormattedTextControl(""), height=1),
        filter=Condition(lambda: chat_state.pending),
    )
    session.layout.container = HSplit([thinking, preview, loading_gap, input_frame])
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
    output.print(Markdown(_indented_lines(text)))


def render_assistant_message(text: str, console: Console | None = None) -> None:
    output = console if console is not None else Console()
    render_markdown(text, output)
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
            elif event.kind == "complete":
                response = state.preview_text
                state.llm.commit_exchange(user_text, response)
                await run_in_terminal(
                    lambda: render_assistant_message(response, console)
                )
                return
            elif event.kind == "error":
                logger.error(
                    "Joy request error: %s",
                    _safe_error(event.error or RuntimeError("unknown provider error")),
                )
                await _render_terminal_notice(
                    app, console, state.preview_text, "Request failed. Try again."
                )
                return
            elif event.kind == "incomplete":
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
        state.pending = False
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
