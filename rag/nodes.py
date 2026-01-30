from __future__ import annotations

from typing import List

from .providers import get_embeddings, get_llm
from .settings import RAGSettings
from .state import AgentState
from .vectorstore import get_vectorstore

_settings = RAGSettings.from_env()


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m"


def _blue(text: str) -> str:
    return f"\033[34m{text}\033[0m"


def _llm_label() -> str:
    provider = (_settings.llm_provider or "ollama").lower()
    if provider == "gemini":
        return f"gemini/{_settings.gemini_llm_model}"
    return f"ollama/{_settings.llm_model}"


def _get_vectorstore():
    embeddings = get_embeddings(_settings)
    return get_vectorstore(settings=_settings, embeddings=embeddings)


def retrieve(state: AgentState):
    print(_blue("---RETRIEVING DOCUMENTS---"))
    question = state["question"]

    vectorstore = _get_vectorstore()
    docs = vectorstore.similarity_search(question, k=_settings.top_k)

    return {"context": [d.page_content for d in docs]}


def _format_prompt(question: str, context: List[str]) -> str:
    joined = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context))
    return (
        "You are a helpful assistant. Use the provided context to answer the question. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{joined}\n\n"
        f"Question: {question}\n"
    )


def generate(state: AgentState):
    print(_blue("---GENERATING ANSWER---"))

    question = state["question"]
    context = state.get("context", [])

    llm = get_llm(_settings)
    prompt = _format_prompt(question=question, context=context)

    print(_yellow(f"[RAG] Using LLM: {_llm_label()}"))
    response = llm.invoke(prompt)
    return {"answer": response.content}
