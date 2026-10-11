"""Terminal chat input and output for joy-agent."""

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.filters import Always, Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    HSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

INPUT_PLACEHOLDER = "Ask Joy to do something…"
DEMO_REPLY = "Agent functionality is not connected yet."
REPLY_DELAY = 1.5
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


@dataclass
class ChatState:
    """Mutable state shared by the prompt layout and reply task."""

    pending: bool = False
    spinner_index: int = 0
    reply_task: asyncio.Task[None] | None = None


def _input_key_bindings(state: ChatState) -> KeyBindings:
    """Create key bindings for submission and cancellation."""

    bindings = KeyBindings()

    @bindings.add("enter", eager=True)
    def submit(event) -> None:
        _submit_buffer(event, state)

    @bindings.add("escape", "enter", eager=True)
    def insert_newline(event) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c", eager=True)
    def cancel_input(event) -> None:
        if state.pending and state.reply_task is not None:
            state.reply_task.cancel()
            return
        event.current_buffer.reset()

    return bindings


def _submit_buffer(event, state: ChatState) -> None:
    """Submit the current buffer or leave it untouched while pending."""

    if state.pending:
        return

    buffer = event.current_buffer
    text = buffer.text
    if not text.strip():
        return
    buffer.reset()

    if text.strip() in {"/quit", "/exit"}:
        event.app.exit()
        return

    state.pending = True
    state.reply_task = event.app.create_background_task(
        _complete_submission(event.app, state, text, event.app.joy_console)
    )
    event.app.invalidate()


def _style() -> Style:
    """Return the true-color style for the input area."""

    return Style.from_dict(
        {
            "frame.border": "fg:#393939 bg:#393939",
            "input": "fg:#ffffff bg:#393939",
            "placeholder": "fg:#999999 bg:#393939",
            "prompt": "fg:#ffffff bg:#393939",
            "thinking": "fg:#aaaaaa",
        }
    )


def create_prompt_session(state: ChatState | None = None) -> PromptSession[str]:
    """Create the fixed-height framed input and loading layout."""

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
    loading = ConditionalContainer(
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
        filter=Condition(lambda: chat_state.pending),
    )
    loading_gap = ConditionalContainer(
        Window(FormattedTextControl(""), height=1),
        filter=Condition(lambda: chat_state.pending),
    )
    session.layout.container = HSplit([loading, loading_gap, input_frame])
    session.joy_state = chat_state
    session.joy_input_frame = input_frame
    return session


def _default_buffer_windows(session: PromptSession[str]) -> Iterable:
    """Find the visible and conditional windows for the default buffer."""

    return (
        window
        for window in session.layout.find_all_windows()
        if isinstance(window.content, BufferControl)
        and window.content.buffer is session.default_buffer
    )


def render_user_message(text: str, console: Console | None = None) -> None:
    """Render a literal user message as a gray, borderless block."""

    output = console if console is not None else Console()
    body = Text(f"› {text}", style="white")
    output.print(
        Panel(
            body,
            box=box.SQUARE,
            border_style="#393939",
            padding=(0, 1),
            style="white on #393939",
            expand=True,
        )
    )
    output.print()


def render_markdown(text: str, console: Console | None = None) -> None:
    """Render Markdown text outside the input area."""

    output = console if console is not None else Console()
    output.print(Markdown(text))


def render_assistant_message(text: str, console: Console | None = None) -> None:
    """Render a labeled assistant response."""

    output = console if console is not None else Console()
    output.print("[bold green]Joy[/bold green]")
    render_markdown(text, output)
    output.print()


async def _animate(app, state: ChatState) -> None:
    """Advance the loading indicator until the reply completes."""

    while state.pending:
        await asyncio.sleep(0.1)
        state.spinner_index = (state.spinner_index + 1) % len(SPINNER_FRAMES)
        app.invalidate()


async def _complete_submission(app, state: ChatState, text: str, console: Console) -> None:
    """Render a submitted message, wait, and render its demo reply."""

    animation = asyncio.create_task(_animate(app, state))
    try:
        await run_in_terminal(lambda: render_user_message(text, console))
        await asyncio.sleep(REPLY_DELAY)
        await run_in_terminal(lambda: render_assistant_message(DEMO_REPLY, console))
    except asyncio.CancelledError:
        raise
    finally:
        animation.cancel()
        state.pending = False
        state.reply_task = None
        app.invalidate()


def run_repl(
    session: PromptSession[str] | None = None,
    console: Console | None = None,
) -> None:
    """Run the terminal chat loop."""

    output = console if console is not None else Console()
    state = ChatState()
    prompt_session = session if session is not None else create_prompt_session(state)
    state = getattr(prompt_session, "joy_state", state)
    prompt_session.app.joy_console = output
    try:
        prompt_session.app.run()
    except (EOFError, KeyboardInterrupt):
        return
