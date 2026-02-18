from __future__ import annotations

from typing import TypedDict, List, Any
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from rag.settings import RAGSettings
from rag.providers import get_llm

class ExtractorState(TypedDict):
    conversation_text: str
    extracted_memories: List[dict[str, Any]]

class MemoryItem(BaseModel):
    category: str = Field(description="The category of the memory (e.g., preference, habit, contact_detail, personal_fact)")
    content: str = Field(description="The concise fact or preference to store.")

class MemoryList(BaseModel):
    items: List[MemoryItem]

def extract_memories(state: ExtractorState):
    settings = RAGSettings.from_env()
    llm = get_llm(settings)
    
    try:
        structured_llm = llm.with_structured_output(MemoryList)
    except NotImplementedError:
        # Fallback for models/providers that might not support it directly in the installed version
        # But commonly used ones (OpenAI, Gemini, Ollama) usually do in recent versions.
        print("[MemoryExtractor] Warning: LLM might not support structured output directly.", flush=True)
        structured_llm = llm

    prompt = (
        "Review the following conversation. Extract any long-term preferences, recurring habits, "
        "or permanent contact details (like the entry for Mumma, or diet restrictions). "
        "Ignore current trip logistics, temporary moods, or simple greetings. "
        "Output a JSON list of items to save."
    )
    
    try:
        response = structured_llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=state["conversation_text"])
        ])
        
        # If response is pydantic object
        if hasattr(response, "items"):
            return {"extracted_memories": [item.model_dump() for item in response.items]}
        # If response is a dict (helper might return dict)
        if isinstance(response, dict) and "items" in response:
             return {"extracted_memories": response["items"]}
             
        # If raw message (fallback)
        print(f"[MemoryExtractor] Got raw response: {response}", flush=True)
        return {"extracted_memories": []}
        
    except Exception as e:
        print(f"[MemoryExtractor] Error during extraction: {e}", flush=True)
        return {"extracted_memories": []}

def build_extractor():
    workflow = StateGraph(ExtractorState)
    workflow.add_node("extract", extract_memories)
    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", END)
    return workflow.compile()
