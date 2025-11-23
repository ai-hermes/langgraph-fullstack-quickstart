import os

from dotenv import load_dotenv
from langchain.messages import AnyMessage
from langchain_core.messages import SystemMessage
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


class MessagesState(BaseModel):
    messages: list[AnyMessage] = []
    llm_calls: int = 0


def llm_call(state: MessagesState):
    return {
        "messages": [
            model.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant"
                    )
                ]
                + state.messages
            )
        ],
        "llm_calls": state.llm_calls + 1
    }


builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)

builder.add_edge(START, "llm_call")
builder.add_edge("llm_call", END)

graph = builder.compile()
