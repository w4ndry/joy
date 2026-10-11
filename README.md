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

Set up the OpenAI provider before running Joy:

```bash
uv sync --extra dev
export JOY_AGENT_API_KEY="your-api-key"
uv run joy
```

You can copy `.env.example` to `.env`. Joy loads the project-local `.env`
automatically, and explicitly exported shell variables take precedence. The
real `.env` file is ignored by Git.

The model and endpoint can also be configured through environment variables:

```bash
export JOY_AGENT_MODEL="openai:gpt-5.6-luna"
export JOY_AGENT_BASE_URL="https://ai.sumopod.com/v1"
uv run joy
```

Command-line options override these environment variables.

The default model is `openai:gpt-5.6-luna`. Select another provider-qualified
model at runtime with:

```bash
uv run joy --model openai:gpt-4o
```

Joy uses the generic `JOY_AGENT_API_KEY` for every provider. Provider-specific
variables such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` remain supported as
fallbacks. Install the matching AISuite provider extra when needed.

For troubleshooting, enable diagnostic logs with `uv run joy --verbose`.
Normal runs do not show diagnostic logs, and API keys are never logged.

OpenAI-compatible providers can use their own endpoint with `--base-url`. For
example, SumoPod can be run with:

```bash
export JOY_AGENT_API_KEY="your-sumopod-key"
uv run joy \
  --model openai:your-sumopod-model \
  --base-url https://ai.sumopod.com/v1
```

Joy streams responses from the selected model. Submitted messages remain in
terminal scrollback, and the next draft can be edited while a response streams.
Ctrl+C cancels UI delivery while preserving the draft; the in-flight request
may continue in the background until its next chunk or the 60-second timeout.

Joy does not read or edit files, execute tests, or persist conversations yet.
The API key is read from the environment and is never stored by Joy.

## Development

```bash
uv run pytest
uv run ruff check .
```
