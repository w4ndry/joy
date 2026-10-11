"""Environment-backed application configuration."""

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration needed to select an LLM provider and model."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load settings from environment variables."""

        defaults = cls()
        provider = os.getenv("JOY_AGENT_PROVIDER", defaults.provider).strip()
        model = os.getenv("JOY_AGENT_MODEL", defaults.model).strip()
        if not provider:
            raise ValueError("JOY_AGENT_PROVIDER must not be empty")
        if not model:
            raise ValueError("JOY_AGENT_MODEL must not be empty")
        return cls(provider=provider, model=model)
