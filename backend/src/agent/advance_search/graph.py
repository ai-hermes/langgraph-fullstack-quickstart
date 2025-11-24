import os

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send
from pydantic import SecretStr

from agent.advance_search.prompts import get_current_date, query_writer_instructions, reflection_instructions, \
    answer_instructions
from agent.advance_search.state import QueryGenerationState, WebSearchState, OverallState, ReflectionState
from agent.advance_search.tools_and_schemas import SearchQueryList, Reflection
from agent.advance_search.utils import get_research_topic

load_dotenv()

from langfuse import get_client

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")


llm = ChatOpenAI(
    base_url=os.environ.get("LLM_API_URL"),
    api_key=SecretStr(os.environ.get("LLM_API_KEY")),
    model=os.environ.get("LLM_API_MODEL"),
    temperature=0,
)


def generate_query(state: OverallState) -> QueryGenerationState:
    # check for custom initial search query count
    structured_llm = llm.with_structured_output(SearchQueryList, method="json_mode")

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state.messages),
        number_queries=state.initial_search_query_count,
    )
    # Generate the search queries
    result: SearchQueryList = structured_llm.invoke(formatted_prompt)
    return QueryGenerationState(
        search_query=result.query
    )


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send(
            "web_research",
            # {"search_query": search_query, "id": int(idx)}
            WebSearchState(search_query=search_query, id=str(idx))
        )
        for idx, search_query in enumerate(state.search_query)
    ]


def web_research(state: WebSearchState) -> OverallState:
    # search with state["search_query"]
    results = DDGS().text(state.search_query, max_results=5)
    """
    {"title": "Python (programming language)", "href": "https://en.wikipedia.org/wiki/Python_(programming_language)", "body": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically type-checked and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming.Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language. Python 3.0, released in 2008, was a major revision and not completely backward-compatible with earlier versions. Beginning with Python 3.5, capabilities and keywords for typing were added to the language, allowing optional static typing. Currently only versions in the 3.x series are supported. Python has gained widespread use in the machine learning community. It is widely taught as an introductory programming language. Since 2003, Python has consistently ranked in the top ten of the most popular programming languages in the TIOBE Programming Community Index, which ranks based on searches in 24 platforms."
    """

    return OverallState(
        sources_gathered=[],
        search_query=[state.search_query],
        web_research_result=[item.get('body') for item in results],
    )


def reflection(state: OverallState) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    # Increment the research loop count and get the reasoning model
    state.research_loop_count = (state.research_loop_count or 0) + 1

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state.messages),
        summaries="\n\n---\n\n".join(state.web_research_result),
    )
    # init Reasoning Model
    result: Reflection = llm.with_structured_output(Reflection, method="json_mode").invoke(formatted_prompt)

    return ReflectionState(
        is_sufficient=result.is_sufficient,
        knowledge_gap=result.knowledge_gap,
        follow_up_queries=result.follow_up_queries,
        research_loop_count=state.research_loop_count,
        number_of_ran_queries=len(state.search_query),
    )


def evaluate_research(state: ReflectionState):
    max_research_loops = (
        state.max_research_loops
        if state.max_research_loops is not None
        else 10
    )
    if state.is_sufficient or state.research_loop_count >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                # {"search_query": follow_up_query,"id": state.number_of_ran_queries + int(idx)},
                WebSearchState(
                    search_query=follow_up_query,
                    id=str(state.number_of_ran_queries + int(idx))
                )
            )
            for idx, follow_up_query in enumerate(state.follow_up_queries)
        ]


def finalize_answer(state: OverallState):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state.messages),
        summaries="\n---\n\n".join(state.web_research_result),
    )

    # init Reasoning Model, default to Gemini 2.5 Flash
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state.sources_gathered:
        if source.short_url in result.content:
            result.content = result.content.replace(
                source.short_url, source.value
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


builder = StateGraph(OverallState)

builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()

graph = builder.compile().with_config({"callbacks": [langfuse_handler]})
