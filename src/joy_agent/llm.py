"""OpenAI streaming integration isolated from the terminal UI."""

from __future__ import annotations

import asyncio
import inspect
import os
import queue
import logging
import threading
from dataclasses import dataclass
from typing import Any, Literal

import aisuite as ai

logger = logging.getLogger(__name__)

MODEL = "openai:gpt-5.6-luna"
REQUEST_TIMEOUT = 60
SYSTEM_PROMPT = (
    "You are Joy, a concise coding assistant. "
    "File reading, file editing, and test execution are not available yet. "
    "Do not claim to have performed those actions."
)


class MissingCredentialsError(RuntimeError):
    """Raised when the selected provider credentials are not configured."""


EventKind = Literal["text", "refusal", "complete", "error", "incomplete"]


@dataclass
class StreamEvent:
    """One event delivered from the synchronous worker to the async UI."""

    generation_id: int
    kind: EventKind
    content: str = ""
    error: Exception | None = None


class StreamHandle:
    """Thread-safe handle for a single synchronous stream."""

    def __init__(self, generation_id: int) -> None:
        self.generation_id = generation_id
        self.events: queue.Queue[StreamEvent] = queue.Queue()
        self.cancel_event = threading.Event()
        self.thread: threading.Thread | None = None

    def cancel(self) -> None:
        """Stop delivering events from this stream to the UI."""

        self.cancel_event.set()


class StreamingLLM:
    """Provider-neutral AISuite client with worker-thread streaming."""

    def __init__(
        self,
        model: str = MODEL,
        base_url: str | None = None,
        client_factory: Any = ai.Client,
    ) -> None:
        provider = self._provider_name(model)
        credential_env = self._credential_env(provider)
        api_key = os.getenv("JOY_AGENT_API_KEY", "").strip()
        if not api_key:
            api_key = os.getenv(credential_env, "").strip()
        if not api_key:
            raise MissingCredentialsError(
                "JOY_AGENT_API_KEY is missing. Set it before running Joy."
            )

        provider_config = {
            "api_key": api_key,
            "timeout": REQUEST_TIMEOUT,
            "max_retries": 0,
        }
        if base_url:
            provider_config["base_url"] = base_url
        self.client = client_factory(
            provider_configs={provider: provider_config}
        )
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.history: list[dict[str, str]] = []
        self._closed = False
        logger.info(
            "configured provider=%s model=%s base_url=%s credential_configured=%s",
            provider,
            model,
            base_url or "provider default",
            bool(api_key),
        )

    @staticmethod
    def _provider_name(model: str) -> str:
        if ":" not in model:
            raise ValueError("Model must use the provider:model format.")
        return model.split(":", 1)[0]

    @staticmethod
    def _credential_env(provider: str) -> str:
        known = {
            "aws-bedrock": "AWS_ACCESS_KEY_ID",
            "azure": "AZURE_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        return known.get(provider, f"{provider.upper().replace('-', '_')}_API_KEY")

    def messages_for(self, user_text: str) -> list[dict[str, str]]:
        """Build a request without committing the pending user message."""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[message.copy() for message in self.history],
            {"role": "user", "content": user_text},
        ]

    def commit_exchange(self, user_text: str, assistant_text: str) -> None:
        """Commit only a successfully completed exchange."""

        self.history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )

    def start_stream(
        self,
        messages: list[dict[str, str]],
        generation_id: int,
    ) -> StreamHandle:
        """Start a synchronous AISuite stream in a daemon worker."""

        handle = StreamHandle(generation_id)
        handle.thread = threading.Thread(
            target=self._stream_worker,
            args=(handle, messages),
            daemon=True,
        )
        handle.thread.start()
        return handle

    def _stream_worker(
        self,
        handle: StreamHandle,
        messages: list[dict[str, str]],
    ) -> None:
        stream = None
        saw_content = False
        completed = False
        try:
            logger.debug(
                "starting stream generation=%s model=%s",
                handle.generation_id,
                self.model,
            )
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                if handle.cancel_event.is_set():
                    return
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None) if delta else None
                refusal = getattr(delta, "refusal", None) if delta else None
                if content:
                    saw_content = True
                    handle.events.put(
                        StreamEvent(handle.generation_id, "text", content=content)
                    )
                if refusal:
                    saw_content = True
                    handle.events.put(
                        StreamEvent(handle.generation_id, "refusal", content=refusal)
                    )
                if getattr(choice, "finish_reason", None) == "stop":
                    completed = True

            if not handle.cancel_event.is_set():
                kind: EventKind = (
                    "complete" if completed and saw_content else "incomplete"
                )
                handle.events.put(StreamEvent(handle.generation_id, kind))
                logger.debug(
                    "stream ended generation=%s status=%s",
                    handle.generation_id,
                    kind,
                )
        except Exception as error:
            if not handle.cancel_event.is_set():
                logger.error(
                    "stream request failed generation=%s error=%s",
                    handle.generation_id,
                    _safe_error(error),
                )
                handle.events.put(
                    StreamEvent(handle.generation_id, "error", error=error)
                )
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    def close(self) -> None:
        """Close the AISuite/OpenAI clients when supported."""

        if self._closed:
            return
        self._closed = True
        _close_client(self.client)
        for provider in getattr(self.client, "providers", {}).values():
            candidates = (
                provider,
                getattr(provider, "client", None),
                getattr(provider, "aclient", None),
            )
            for candidate in candidates:
                _close_client(candidate)


def _close_client(client: Any) -> None:
    """Close sync clients and await async client cleanup from sync code."""

    close = getattr(client, "close", None)
    if not callable(close):
        return
    result = close()
    if not inspect.isawaitable(result):
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(result)
    else:
        asyncio.create_task(result)


def _safe_error(error: Exception) -> str:
    """Remove configured secrets from provider error messages."""

    message = str(error)
    for variable in ("JOY_AGENT_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(variable, "")
        if value:
            message = message.replace(value, "[redacted]")
    return message or type(error).__name__

