from __future__ import annotations

import operator
from dataclasses import dataclass, field

from langchain_core.messages import AnyMessage
from pydantic import BaseModel
from typing_extensions import Annotated


def keep_it(left, right):
    return max(left, right)

class OverallState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add] = []
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: Annotated[int, keep_it] = 3
    max_research_loops:  Annotated[int, keep_it] = 10
    research_loop_count: Annotated[int, keep_it] = 0


class ReflectionState(BaseModel):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: list[str]
    research_loop_count: int
    number_of_ran_queries: int
    max_research_loops: Annotated[int, keep_it] = 10


class QueryGenerationState(BaseModel):
    search_query: list[str]


class WebSearchState(BaseModel):
    search_query: str
    id: str


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
