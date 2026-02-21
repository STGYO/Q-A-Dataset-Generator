"""LLM client abstraction supporting online (OpenAI / Gemini) and offline (Ollama) models."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Encapsulates an LLM completion result with timing metadata."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.latency_seconds > 0 and self.completion_tokens > 0:
            return self.completion_tokens / self.latency_seconds
        return 0.0


# ---------------------------------------------------------------------------
# Base client
# ---------------------------------------------------------------------------


class BaseLLMClient:
    """Abstract base class for LLM clients."""

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI-compatible client (online + Ollama offline)
# ---------------------------------------------------------------------------


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API and any OpenAI-compatible endpoint (e.g. Ollama, LM Studio).

    Args:
        model: Model name (e.g. ``"gpt-4o"``, ``"llama3"``).
        api_key: API key.  For offline endpoints this can be any non-empty string.
        base_url: API base URL.  Defaults to the official OpenAI endpoint.
            For Ollama use ``"http://localhost:11434/v1"``.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key or "ollama"  # Ollama ignores the key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai package is required. Install it with `pip install openai`."
                ) from exc
            kwargs: Dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        latency = time.perf_counter() - start

        choice = response.choices[0]
        usage = response.usage or {}

        return LLMResponse(
            content=choice.message.content or "",
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            latency_seconds=latency,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )


# ---------------------------------------------------------------------------
# Google Gemini client
# ---------------------------------------------------------------------------


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini via the ``google-generativeai`` SDK.

    Args:
        model: Gemini model name (e.g. ``"gemini-1.5-flash"``).
        api_key: Google AI Studio API key.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai package required. "
                "Install it with `pip install google-generativeai`."
            ) from exc

        if self.api_key:
            genai.configure(api_key=self.api_key)

        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        model = genai.GenerativeModel(
            self.model,
            generation_config=generation_config,
        )

        start = time.perf_counter()
        response = model.generate_content(prompt)
        latency = time.perf_counter() - start

        content = response.text if hasattr(response, "text") else ""
        return LLMResponse(
            content=content,
            model=self.model,
            latency_seconds=latency,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm_client(
    mode: str = "online",
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> BaseLLMClient:
    """Factory function to create an LLM client.

    Args:
        mode: ``"online"`` (cloud API) or ``"offline"`` (local model via Ollama/LM Studio).
        provider: ``"openai"``, ``"gemini"``, or ``"ollama"``.
        model: Model identifier.
        api_key: API key for cloud providers.
        base_url: Override the API base URL.  Auto-set to Ollama's default for
            ``mode="offline"`` when *provider* is ``"ollama"``.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        A configured :class:`BaseLLMClient` instance.
    """
    if mode == "offline" or provider == "ollama":
        _base_url = base_url or "http://localhost:11434/v1"
        return OpenAIClient(
            model=model,
            api_key=api_key or "ollama",
            base_url=_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "gemini":
        return GeminiClient(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Default: OpenAI
    return OpenAIClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
