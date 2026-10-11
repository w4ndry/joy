"""Command-line entry point for joy-agent."""

from .config import Settings
from .llm import LLMClient
from .ui import run_repl


def main() -> None:
    """Start the joy-agent command."""

    settings = Settings.from_environment()
    run_repl(LLMClient(settings))


if __name__ == "__main__":
    main()
