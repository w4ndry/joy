# joy-agent

A starter project for a terminal-based coding agent.

The project is intentionally small. It provides a three-row terminal input
for Joy and a Rich output area, ready for future coding-agent behavior.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --extra dev
```

## Run

```bash
uv run joy
```

The input has one blank gray row above and below the editable row. Press Enter
to submit, Alt+Enter to insert a newline, Ctrl+C to cancel the current input,
or Ctrl+D on an empty input to exit. Enter `/quit` or `/exit` to exit.

Submitted messages remain in terminal scrollback. Joy shows a simulated
`Thinking…` indicator for 1.5 seconds, then displays a labeled demo response:
`Agent functionality is not connected yet.` No AI requests or coding tools are
connected yet.

## Development

```bash
uv run pytest
uv run ruff check .
```
