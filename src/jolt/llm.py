"""Language-model backends behind one small interface.

Every agent node talks to an ``LLM`` through ``chat(messages, ...)``. Swapping the local
llama.cpp model for Ollama or a hosted API is a config change, not a code change.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Protocol

from jolt.config import Settings

logger = logging.getLogger(__name__)

Message = dict[str, str]


class LLM(Protocol):
    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str: ...


class LlamaCppLLM:
    """Local GGUF model via llama-cpp-python. Uses the model's own chat template."""

    def __init__(self, settings: Settings) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "llama-cpp-python is not installed. Run: pip install 'jolt[local]'"
            ) from exc
        if not settings.llm_model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {settings.llm_model_path}. "
                "Download a GGUF model or set JOLT_LLM_MODEL_PATH."
            )
        self._llm = Llama(
            model_path=str(settings.llm_model_path),
            n_ctx=settings.llm_n_ctx,
            n_gpu_layers=settings.llm_n_gpu_layers,
            verbose=False,
        )
        self._temperature = settings.llm_temperature
        self._lock = threading.Lock()  # a llama.cpp context is not thread-safe

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self._temperature if temperature is None else temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        with self._lock:
            out = self._llm.create_chat_completion(**kwargs)
        return str(out["choices"][0]["message"]["content"] or "").strip()


class OpenAICompatibleLLM:
    """Any server that speaks the OpenAI chat-completions API (Ollama, vLLM, Groq, OpenAI)."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.llm_timeout_s,
        )
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._json_supported = True

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        from openai import BadRequestError

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self._temperature if temperature is None else temperature,
        }
        if json_mode and self._json_supported:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except BadRequestError:
            if "response_format" not in kwargs:
                raise
            # Some servers reject response_format; fall back to prompt-only JSON.
            logger.info("Server rejected JSON mode; retrying without it")
            self._json_supported = False
            kwargs.pop("response_format")
            resp = self._client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()


def build_llm(settings: Settings) -> LLM:
    if settings.llm_backend == "llamacpp":
        return LlamaCppLLM(settings)
    if settings.llm_backend == "openai":
        return OpenAICompatibleLLM(settings)
    from jolt.testing import ExtractiveFakeLLM

    logger.warning("Using the fake LLM backend: answers are extractive, not generated")
    return ExtractiveFakeLLM()


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    text = text.strip()
    for candidate in (text, *(_JSON_BLOCK.findall(text) or [])):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None
