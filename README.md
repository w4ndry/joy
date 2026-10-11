# joy-agent

A starter project for a terminal-based coding agent.

The project is intentionally small. It provides a terminal REPL that sends
prompts to an LLM and renders responses as Markdown, with the package
boundaries needed for future coding-agent behavior.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --extra dev
```

## Configuration

Set these variables when preparing to connect an LLM provider:

```bash
export JOY_AGENT_PROVIDER=openai
export JOY_AGENT_MODEL=gpt-4o-mini
```

Provider credentials should use the environment variables expected by the
selected provider's SDK, such as `OPENAI_API_KEY`. No credentials are stored
in this repository.

## Run

```bash
uv run joy
```

The REPL accepts prompts until you enter `exit`, `quit`, Ctrl-D, or Ctrl-C.
Conversation history is kept in memory for the current process. The current
version does not execute tools, modify files, stream responses, or persist
conversation history.

## Development

```bash
uv run pytest
uv run ruff check .
```
