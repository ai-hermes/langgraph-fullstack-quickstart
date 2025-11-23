from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated


class OverallState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add] = []
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: Optional[int | None] = None
    max_research_loops:  Optional[int | None] = None
    research_loop_count: Optional[int | None] = None


class ReflectionState(BaseModel):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: list[str]
    research_loop_count: int
    number_of_ran_queries: int
    max_research_loops: Optional[int | None] = None


class QueryGenerationState(BaseModel):
    search_query: list[str]


class WebSearchState(BaseModel):
    search_query: str
    id: str


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
