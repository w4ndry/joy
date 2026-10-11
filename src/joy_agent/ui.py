"""Terminal input and output for joy-agent."""

from collections.abc import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Always
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame
from rich.console import Console
from rich.markdown import Markdown

INPUT_PLACEHOLDER = "Ask Joy to do something…"
AGENT_NOT_CONNECTED = "Agent functionality is not connected yet."


def _input_key_bindings() -> KeyBindings:
    """Create the key bindings for the one-row input."""

    bindings = KeyBindings()

    @bindings.add("enter", eager=True)
    def submit(event) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter", eager=True)
    def insert_newline(event) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c", eager=True)
    def cancel_input(event) -> None:
        event.current_buffer.reset()

    return bindings


def _style() -> Style:
    """Return the true-color style for the input area."""

    return Style.from_dict(
        {
            "frame.border": "fg:#393939 bg:#393939",
            "input": "fg:#ffffff bg:#393939",
            "placeholder": "fg:#999999 bg:#393939",
            "prompt": "fg:#ffffff bg:#393939",
        }
    )


def create_prompt_session() -> PromptSession[str]:
    """Create the fixed-height, framed prompt session."""

    session = PromptSession(
        message=[("class:prompt", "› ")],
        multiline=True,
        placeholder=[("class:placeholder", INPUT_PLACEHOLDER)],
        key_bindings=_input_key_bindings(),
        style=_style(),
        color_depth=ColorDepth.TRUE_COLOR,
    )

    for window in _default_buffer_windows(session):
        window.height = Dimension.exact(1)
        window.dont_extend_height = Always()
        window.style = "class:input"

    # Frame supplies the two hidden one-row borders around the existing
    # PromptSession layout. No padding or Box wrapper is needed.
    session.layout.container = Frame(
        session.layout.container,
        height=Dimension.exact(3),
    ).container
    return session


def _default_buffer_windows(session: PromptSession[str]) -> Iterable:
    """Find the visible and conditional windows for the default buffer."""

    return (
        window
        for window in session.layout.find_all_windows()
        if isinstance(window.content, BufferControl)
        and window.content.buffer is session.default_buffer
    )


def render_markdown(text: str, console: Console | None = None) -> None:
    """Render Markdown text outside the input area."""

    output = console if console is not None else Console()
    output.print(Markdown(text))


def run_repl(
    session: PromptSession[str] | None = None,
    console: Console | None = None,
) -> None:
    """Run the terminal input loop."""

    prompt_session = session if session is not None else create_prompt_session()
    output = console if console is not None else Console()

    while True:
        try:
            user_input = prompt_session.prompt().strip()
        except (EOFError, KeyboardInterrupt):
            return

        if user_input in {"/quit", "/exit"}:
            return
        if not user_input:
            continue

        output.print(AGENT_NOT_CONNECTED)
