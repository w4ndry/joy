"""Command-line entry point for joy."""

import argparse
import logging
import os
from pathlib import Path

from rich.console import Console

from .llm import MODEL, MissingCredentialsError, StreamingLLM
from .ui import run_repl


def _load_project_env() -> None:
    """Load simple KEY=VALUE entries from the project-local .env file."""

    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if not separator or not name or not name.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Joy coding assistant.")
    parser.add_argument(
        "--model",
        default=os.getenv("JOY_AGENT_MODEL", os.getenv("JOY_MODEL", MODEL)),
        help=f"Provider-qualified model name (default: {MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "JOY_AGENT_BASE_URL", os.getenv("JOY_BASE_URL")
        ),
        help="Optional provider API base URL, for OpenAI-compatible endpoints.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable diagnostic logs.",
    )
    return parser.parse_args()


def main() -> None:
    """Start the joy command."""

    # Explicit shell variables win over values from the project-local .env.
    _load_project_env()
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    console = Console()
    try:
        llm = StreamingLLM(model=args.model, base_url=args.base_url)
    except MissingCredentialsError as error:
        console.print(f"[red]Setup required:[/red] {error}")
        return
    run_repl(llm, console=console)


if __name__ == "__main__":
    main()
