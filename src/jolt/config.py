"""Runtime configuration, read from environment variables (prefix ``JOLT_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JOLT_", env_file=".env", extra="ignore")

    # --- Language model ---------------------------------------------------------------------
    # "llamacpp": local GGUF model on your GPU/CPU.
    # "openai":   any OpenAI-compatible server (Ollama, vLLM, LM Studio, Groq, OpenAI).
    # "fake":     deterministic offline stand-in used for tests, CI and smoke runs.
    llm_backend: Literal["llamacpp", "openai", "fake"] = "llamacpp"
    llm_model_path: Path = Path("models/qwen2.5-3b-instruct-q4_k_m.gguf")
    llm_n_ctx: int = Field(4096, ge=512)
    llm_n_gpu_layers: int = -1  # -1 offloads every layer to the GPU
    llm_model: str = "qwen2.5:3b"  # model name for the OpenAI-compatible backend
    openai_base_url: str | None = "http://localhost:11434/v1"  # Ollama default
    openai_api_key: SecretStr = SecretStr("not-needed")
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    llm_timeout_s: float = Field(120.0, gt=0)

    # --- Embeddings and reranking ------------------------------------------------------------
    embedding_backend: Literal["sentence-transformers", "openai", "hashing"] = (
        "sentence-transformers"
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_backend: Literal["cross-encoder", "none"] = "cross-encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Index and chunking ------------------------------------------------------------------
    index_dir: Path = Path("data/index")
    upload_dir: Path = Path("data/uploads")
    chunk_size: int = Field(800, ge=100, description="Target chunk size in characters")
    chunk_overlap: int = Field(120, ge=0)

    # --- Agent behaviour ---------------------------------------------------------------------
    retrieve_k: int = Field(20, ge=1, description="Candidates fetched by each retriever")
    rerank_top_n: int = Field(5, ge=1, description="Chunks kept after reranking")
    max_rewrites: int = Field(2, ge=0, description="Query rewrites before falling back")
    web_search_enabled: bool = True
    web_max_results: int = Field(4, ge=1, le=10)
    learn_from_web: bool = False  # add useful web results to the local index

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
