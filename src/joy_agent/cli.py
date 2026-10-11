"""Command-line entry point for joy."""

from .ui import run_repl


def main() -> None:
    """Start the joy command."""

    run_repl()


if __name__ == "__main__":
    main()
