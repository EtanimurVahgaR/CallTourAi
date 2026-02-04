from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .nodes import generate, retrieve
from .state import AgentState


def build_app():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_node", retrieve)
    workflow.add_node("generate_node", generate)

    workflow.add_edge(START, "retrieve_node")
    workflow.add_edge("retrieve_node", "generate_node")
    workflow.add_edge("generate_node", END)

    return workflow.compile()


@lru_cache(maxsize=1)
def get_app():
    return build_app()
