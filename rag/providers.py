from __future__ import annotations

import importlib

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from rag.settings import RAGSettings


def _strip_models_prefix(model_name: str) -> str:
    name = (model_name or "").strip()
    if name.startswith("models/"):
        return name[len("models/"):]
    return name


def get_embeddings(settings: RAGSettings) -> Embeddings:
    provider = (settings.embed_provider or "ollama").lower()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

    if provider == "gemini":
        try:
            module = importlib.import_module("langchain_google_genai")
            GoogleGenerativeAIEmbeddings = getattr(module, "GoogleGenerativeAIEmbeddings")
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Gemini provider selected but langchain-google-genai is not installed. "
                "Install it with: pip install langchain-google-genai"
            ) from e

        # API key is read from GOOGLE_API_KEY
        return GoogleGenerativeAIEmbeddings(model=_strip_models_prefix(settings.gemini_embedding_model))

    raise ValueError(f"Unknown embed provider: {provider}")


def get_llm(settings: RAGSettings) -> BaseChatModel:
    provider = (settings.llm_provider or "ollama").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
        )

    if provider == "gemini":
        try:
            module = importlib.import_module("langchain_google_genai")
            ChatGoogleGenerativeAI = getattr(module, "ChatGoogleGenerativeAI")
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Gemini provider selected but langchain-google-genai is not installed. "
                "Install it with: pip install langchain-google-genai"
            ) from e

        # API key is read from GOOGLE_API_KEY
        try:
            return ChatGoogleGenerativeAI(model=_strip_models_prefix(settings.gemini_llm_model))
        except Exception as e:
            # Surface a more helpful hint for the common "model not found" error.
            msg = str(e)
            if "NOT_FOUND" in msg or "not found" in msg.lower():
                raise RuntimeError(
                    "Gemini model was not found for your current API/version. "
                    "Try setting GEMINI_LLM_MODEL to a supported name, e.g. 'gemini-flash-latest' (no 'models/' prefix), "
                    "or check the available models in your Google AI Studio account."
                ) from e
            raise

    raise ValueError(f"Unknown LLM provider: {provider}")
