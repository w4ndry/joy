# AGENTS.md

## Project overview

`joy-agent` (the `joy` command) is a small terminal-based coding-agent
scaffold. It currently provides a Rich/prompt-toolkit chat UI and streams model
responses through AISuite. It is not yet a full coding agent: it does not read
or edit files, execute user tests, or persist conversations.

The repository is a Python package using a `src/` layout:

```text
src/joy_agent/
  cli.py       # command-line parsing, .env loading, application startup
  llm.py       # provider-neutral AISuite client and streaming worker
  ui.py        # prompt-toolkit UI, streaming state, and Rich rendering
tests/
  test_smoke.py
```

## Runtime and infrastructure

- Requires Python 3.11 or newer.
- Use `uv` for environments, dependency installation, and command execution.
- Build backend: Hatchling. The package is built from `src/joy_agent`.
- Runtime dependencies are AISuite (with the OpenAI extra), prompt-toolkit,
  and Rich. Development dependencies are pytest and Ruff.
- There is no database, container, cloud deployment, local service, or checked-in
  CI workflow. The only external runtime dependency is the configured model API.
- `uv.lock` is committed; update it when dependency declarations change.
- `.env` is project-local and ignored by Git. Never commit credentials or print
  API keys in logs, errors, test output, or documentation.

## Setup and common commands

From the repository root:

```bash
uv sync --extra dev
uv run joy
uv run pytest
uv run ruff check .
```

For a real provider run, set `JOY_AGENT_API_KEY` (or copy `.env.example` to
`.env`) before starting Joy. The model and optional OpenAI-compatible endpoint
are configured with `JOY_AGENT_MODEL` and `JOY_AGENT_BASE_URL`, or overridden
with `--model` and `--base-url`. Use `uv run joy --verbose` for diagnostics.
Tests should not require credentials or network access.

## Code conventions

- Follow the existing Python style: four spaces, type annotations for public
  interfaces, short docstrings, and a line length of 88 characters.
- Keep provider/API concerns in `llm.py`, terminal layout and rendering in
  `ui.py`, and process startup/argument handling in `cli.py`.
- Prefer small, testable functions and dependency injection/fakes over live API
  calls in tests.
- Preserve the `src/joy_agent` import layout and the `joy_agent.cli:main`
  console entry point.
- Use standard-library logging for diagnostics. Never log credential values or
  include them in exception messages; use the existing redaction behavior in
  `llm.py` when handling provider errors.
- Preserve the provider-qualified model format (`provider:model`). Generic
  `JOY_AGENT_API_KEY` takes precedence over provider-specific fallback keys.
- Keep the UI responsive: provider streaming runs in a worker thread and UI
  delivery runs asynchronously. Preserve generation IDs, cancellation behavior,
  and client/stream cleanup when changing this flow.
- Keep conversation history transactional: only commit a user/assistant pair
  after a stream completes successfully.

## Formatting, linting, and tests

Ruff is configured in `pyproject.toml` with an 88-character line length,
Python 3.11 target, and rules `E`, `F`, and `I` (including import sorting).
Run both checks before handing off code:

```bash
uv run ruff check .
uv run pytest
```

Tests live in `tests/test_smoke.py` and cover credentials, streaming events,
history, cleanup, prompt layout, rendering, submission, and cancellation.
When changing behavior, add or update focused tests using the existing fake
client/stream patterns. Do not make tests depend on a live model, API key, or
network availability.

## Configuration and safety notes

- `.env` is loaded from the current working directory, and existing shell
  variables win over values from that file.
- The default model is `openai:gpt-5.6-luna`; model names must include a
  provider prefix.
- API requests use a 60-second timeout and no retries by default.
- Ctrl+C while a response is pending cancels UI delivery and preserves the
  current draft; do not casually change this user-visible behavior.
- `/quit`, `/exit`, Ctrl+D on empty input, and normal keyboard interruption are
  exit paths. Ensure the AISuite client is closed on every exit path.
- The current system prompt explicitly says file reading/editing and test
  execution are unavailable. Update that contract if capabilities are added.

## Change guidance

Before modifying behavior, read the relevant module and its tests. Keep changes
focused, update README instructions when commands or configuration change, and
run Ruff plus pytest. If adding a dependency, edit `pyproject.toml` and refresh
`uv.lock` with uv; do not hand-edit lockfile package data.

