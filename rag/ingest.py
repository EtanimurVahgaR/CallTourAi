from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .providers import get_embeddings
from .vectorstore import get_vectorstore

from .settings import RAGSettings


def ingest(data_dir: str, chroma_dir: str) -> int:
    settings = RAGSettings.from_env()

    # CLI --chroma only applies to local mode
    if (settings.chroma_mode or "local").lower() == "local":
        settings = RAGSettings(
            llm_provider=settings.llm_provider,
            embed_provider=settings.embed_provider,
            chroma_mode=settings.chroma_mode,
            chroma_dir=chroma_dir,
            chroma_http_host=settings.chroma_http_host,
            chroma_http_port=settings.chroma_http_port,
            chroma_http_ssl=settings.chroma_http_ssl,
            chroma_http_api_key=settings.chroma_http_api_key,
            chroma_tenant=settings.chroma_tenant,
            chroma_database=settings.chroma_database,
            collection_name=settings.collection_name,
            embedding_model=settings.embedding_model,
            llm_model=settings.llm_model,
            gemini_embedding_model=settings.gemini_embedding_model,
            gemini_llm_model=settings.gemini_llm_model,
            top_k=settings.top_k,
            ollama_base_url=settings.ollama_base_url,
        )

    base = Path(data_dir)
    if not base.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    files = sorted([p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md"}])
    if not files:
        raise SystemExit(
            f"No .txt/.md files found in {data_dir}. Put some files in ./data first."
        )

    docs = []
    for fp in files:
        loader = TextLoader(str(fp), encoding="utf-8")
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings(settings)

    # Create (or update) DB (local or remote depending on CHROMA_MODE)
    vectorstore = get_vectorstore(settings=settings, embeddings=embeddings)
    vectorstore.add_documents(chunks)

    # Only local persistence uses persist_directory.
    if (settings.chroma_mode or "local").lower() == "local":
        persist = getattr(vectorstore, "persist", None)
        if callable(persist):
            persist()

    print(f"Ingested {len(files)} files -> {len(chunks)} chunks")
    if (settings.chroma_mode or "local").lower() == "local":
        print(f"Chroma persisted to: {settings.chroma_dir}")
    else:
        print(
            "Chroma stored remotely via HTTP: "
            f"{settings.chroma_http_host}:{settings.chroma_http_port} (ssl={settings.chroma_http_ssl})"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest ./data into a Chroma vector DB")
    parser.add_argument("--data", default="./data", help="Folder with .txt/.md files")
    parser.add_argument("--chroma", default="./chroma_db", help="Chroma persist directory")
    args = parser.parse_args()

    return ingest(data_dir=args.data, chroma_dir=args.chroma)


if __name__ == "__main__":
    raise SystemExit(main())
