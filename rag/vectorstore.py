from __future__ import annotations

from typing import Any, Optional

import chromadb
from langchain_core.embeddings import Embeddings

from langchain_chroma import Chroma

from rag.settings import RAGSettings


def _build_http_client(settings: RAGSettings):
    if not settings.chroma_http_host:
        raise ValueError("CHROMA_MODE=http requires CHROMA_HTTP_HOST (or CHROMA_HOST)")

    headers: Optional[dict[str, str]] = None
    if settings.chroma_http_api_key:
        # Common pattern for Chroma Cloud / secured proxies.
        headers = {"Authorization": f"Bearer {settings.chroma_http_api_key}"}

    kwargs: dict[str, Any] = {
        "host": settings.chroma_http_host,
        "port": settings.chroma_http_port,
        "ssl": settings.chroma_http_ssl,
    }
    if headers:
        kwargs["headers"] = headers

    # Some Chroma deployments support multi-tenancy; keep this best-effort.
    if settings.chroma_tenant:
        kwargs["tenant"] = settings.chroma_tenant
    if settings.chroma_database:
        kwargs["database"] = settings.chroma_database

    try:
        return chromadb.HttpClient(**kwargs)
    except TypeError:
        # Older chromadb versions may not accept tenant/database.
        kwargs.pop("tenant", None)
        kwargs.pop("database", None)
        return chromadb.HttpClient(**kwargs)


def _build_cloud_client(settings: RAGSettings):
    cloud_client = getattr(chromadb, "CloudClient", None)
    if cloud_client is None:
        raise RuntimeError(
            "chromadb.CloudClient is not available in your installed chromadb version. "
            "Upgrade chromadb to use Chroma Cloud."
        )

    api_key = settings.chroma_http_api_key
    tenant = settings.chroma_tenant
    database = settings.chroma_database
    if not api_key:
        raise ValueError("Chroma Cloud requires CHROMA_API_KEY")
    if not tenant:
        raise ValueError("Chroma Cloud requires CHROMA_TENANT")
    if not database:
        raise ValueError("Chroma Cloud requires CHROMA_DATABASE")

    return cloud_client(api_key=api_key, tenant=tenant, database=database)


def get_vectorstore(settings: RAGSettings, embeddings: Embeddings) -> Chroma:
    mode = (settings.chroma_mode or "local").lower()

    if mode == "local":
        return Chroma(
            collection_name=settings.collection_name,
            persist_directory=settings.chroma_dir,
            embedding_function=embeddings,
        )

    if mode == "http":
        # If you're pointing at Chroma Cloud (trychroma.com), use CloudClient.
        if settings.chroma_http_host and "trychroma.com" in settings.chroma_http_host:
            client = _build_cloud_client(settings)
        else:
            client = _build_http_client(settings)
        return Chroma(
            client=client,
            collection_name=settings.collection_name,
            embedding_function=embeddings,
        )

    if mode == "cloud":
        client = _build_cloud_client(settings)
        return Chroma(
            client=client,
            collection_name=settings.collection_name,
            embedding_function=embeddings,
        )

    raise ValueError(f"Unknown CHROMA_MODE: {settings.chroma_mode}")
