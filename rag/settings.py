from __future__ import annotations

from dataclasses import dataclass
import os

# Load .env automatically so users don't need to export variables manually.
# If python-dotenv isn't installed, we simply skip loading.
try:  # pragma: no cover
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:  # pragma: no cover
    pass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value
    return None


@dataclass(frozen=True)
class RAGSettings:
    # Providers
    llm_provider: str = "ollama"  # ollama | gemini
    embed_provider: str = "ollama"  # ollama | gemini

    # Vector store (Chroma)
    # - local: persist to CHROMA_DIR on disk
    # - http: connect to a remote Chroma server / Chroma Cloud over HTTP
    chroma_mode: str = "local"  # local | http
    chroma_dir: str = "./chroma_db"
    chroma_http_host: str | None = None
    chroma_http_port: int = 8000
    chroma_http_ssl: bool = False
    chroma_http_api_key: str | None = None
    chroma_tenant: str | None = None
    chroma_database: str | None = None
    collection_name: str = "calltourai"
    # Ollama models
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"

    # Gemini models (used when provider=gemini)
    gemini_embedding_model: str = "text-embedding-004"
    # Many SDKs/providers expose the "-latest" alias more reliably than the bare name.
    gemini_llm_model: str = "gemini-flash-latest"
    top_k: int = 3

    # Optional: point to a non-default Ollama server (e.g. http://localhost:11434)
    ollama_base_url: str | None = None

    # Optional: custom base URL is not used for Gemini via langchain-google-genai

    @staticmethod
    def from_env() -> "RAGSettings":
        # Back-compat + shorter aliases:
        # - CHROMA_HOST == CHROMA_HTTP_HOST
        # - CHROMA_API_KEY == CHROMA_HTTP_API_KEY
        chroma_host = _env_first("CHROMA_HTTP_HOST", "CHROMA_HOST")

        # If CHROMA_HOST is set, assume a hosted endpoint unless overridden.
        default_port = 443 if chroma_host else 8000
        default_ssl = True if chroma_host else False

        return RAGSettings(
            llm_provider=(os.getenv("RAG_LLM_PROVIDER", "ollama").strip().lower() or "ollama"),
            embed_provider=(os.getenv("RAG_EMBED_PROVIDER", "ollama").strip().lower() or "ollama"),
            chroma_mode=(os.getenv("CHROMA_MODE", "local").strip().lower() or "local"),
            chroma_dir=os.getenv("CHROMA_DIR", "./chroma_db"),
            chroma_http_host=chroma_host,
            chroma_http_port=_env_int("CHROMA_HTTP_PORT", _env_int("CHROMA_PORT", default_port)),
            chroma_http_ssl=_env_bool("CHROMA_HTTP_SSL", _env_bool("CHROMA_SSL", default_ssl)),
            chroma_http_api_key=_env_first("CHROMA_HTTP_API_KEY", "CHROMA_API_KEY"),
            chroma_tenant=os.getenv("CHROMA_TENANT") or None,
            chroma_database=os.getenv("CHROMA_DATABASE") or None,
            collection_name=os.getenv("CHROMA_COLLECTION", "calltourai"),
            embedding_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            llm_model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
            gemini_embedding_model=os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004"),
            gemini_llm_model=os.getenv("GEMINI_LLM_MODEL", "gemini-flash-latest"),
            top_k=_env_int("RAG_TOP_K", 3),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL") or None,
        )
