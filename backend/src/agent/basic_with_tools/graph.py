import os

from dotenv import load_dotenv
from langchain.messages import AnyMessage
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import SecretStr, BaseModel

load_dotenv()

model = ChatOpenAI(
    base_url=os.environ.get("LLM_API_URL"),
    api_key=SecretStr(os.environ.get("LLM_API_KEY")),
    model=os.environ.get("LLM_API_MODEL"),
    temperature=0,
)


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


class MessagesState(BaseModel):
    messages: list[AnyMessage] = []
    llm_calls: int = 0


def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return MessagesState(
        messages= [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state.messages
            )
        ],
        llm_calls = state.llm_calls + 1,
    )


def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state.messages[-1].tool_calls: # noqa
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState):
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state.messages
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


builder = StateGraph(MessagesState)

builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")
builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
builder.add_edge("tool_node", "llm_call")

graph = builder.compile()
