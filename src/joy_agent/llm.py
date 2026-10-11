"""LLM integration boundary.

Provider-specific credentials are intentionally handled by the provider SDK.
This module keeps AISuite usage isolated from the terminal UI.
"""

import aisuite as ai

from .config import Settings


class LLMClient:
    """Small AISuite adapter for the terminal agent."""

    def __init__(self, settings: Settings) -> None:
        self._client = ai.Client()
        self._model = model_name(settings)
        self._messages: list[dict[str, str]] = []

    @property
    def messages(self) -> list[dict[str, str]]:
        """Return the current in-memory conversation history."""

        return self._messages.copy()

    def send(self, user_input: str) -> str:
        """Send a user message and return the assistant's response."""

        self._messages.append({"role": "user", "content": user_input})
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The provider returned an empty response.")
        self._messages.append({"role": "assistant", "content": content})
        return content


def model_name(settings: Settings) -> str:
    """Return the provider-qualified model name expected by AISuite."""

    return f"{settings.provider}:{settings.model}"
