from __future__ import annotations

from typing import List, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    context: List[str]
    answer: str
