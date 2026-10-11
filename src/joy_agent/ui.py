"""Terminal UI integration boundary."""

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.markdown import Markdown

from .llm import LLMClient


def create_prompt_session() -> PromptSession[str]:
    """Create the interactive prompt session for future agent input."""

    return PromptSession()


def render_markdown(text: str, console: Console | None = None) -> None:
    """Render Markdown text to the terminal."""

    (console if console is not None else Console()).print(Markdown(text))


def run_repl(
    llm_client: LLMClient,
    session: PromptSession[str] | None = None,
    console: Console | None = None,
) -> None:
    """Run the interactive terminal loop."""

    prompt_session = session or create_prompt_session()
    output = console if console is not None else Console()
    output.print("joy-agent ready. Type 'exit' or 'quit' to leave.")

    while True:
        try:
            user_input = prompt_session.prompt("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            output.print("\\nGoodbye!")
            return

        if user_input.lower() in {"exit", "quit"}:
            output.print("Goodbye!")
            return
        if not user_input:
            continue

        try:
            response = llm_client.send(user_input)
        except Exception:
            output.print(
                "[red]Request failed.[/red] Check your provider configuration "
                "and try again."
            )
            continue
        render_markdown(response, output)
