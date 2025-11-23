import operator
import os

from dotenv import load_dotenv
from langchain.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from pydantic import SecretStr
from typing_extensions import TypedDict, Annotated

from backend.src.advance_search.prompts import get_current_date, query_writer_instructions, web_searcher_instructions
from backend.src.advance_search.state import QueryGenerationState, WebSearchState, OverallState
from backend.src.advance_search.tools_and_schemas import SearchQueryList
from backend.src.advance_search.utils import get_research_topic

load_dotenv()

llm = ChatOpenAI(
    base_url=os.environ.get("LLM_API_URL"),
    api_key=SecretStr(os.environ.get("LLM_API_KEY")),
    model=os.environ.get("LLM_API_MODEL"),
    temperature=0,
)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def generate_query(state: dict) -> QueryGenerationState:
    structured_llm = llm.with_structured_output(SearchQueryList)
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}

def web_research(state: WebSearchState) -> OverallState:
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

builder = StateGraph(MessagesState)

builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)

graph = builder.compile()
