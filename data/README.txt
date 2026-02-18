Put your knowledge base files here.

- Add one or more .txt files
- Ingest into Chroma: python -m rag.ingest
- Ask via CLI: python -m rag.run_rag "..."

Notes:
- The vector store is controlled by .env (CHROMA_MODE + CHROMA_* settings).
- Local mode will persist under CHROMA_DIR; HTTP/Cloud modes write to your remote Chroma.
