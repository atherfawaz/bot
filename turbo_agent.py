import streamlit as st

# from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.memory import StreamlitChatMessageHistory
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAI

# load_dotenv()

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"
history = StreamlitChatMessageHistory(key="st_history_key")


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PurchaseInput(BaseModel):
    query: str = Field(description="should be the name of a product to buy")


@st.cache_resource
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        verbose=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )


@st.cache_resource(ttl="1h")
def get_retriever():
    vectorstore = Pinecone.from_existing_index(
        "catalog-768",
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
        "text",
    )
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    return retriever


@st.cache_resource
def get_tool_llm():
    return OpenAI(temperature=0)


# @st.cache_resource
# def get_prebuilt_agents():
#     return load_tools(["llm-math"], llm=get_tool_llm())


@tool("buy-product", args_schema=PurchaseInput)
def buy_product(query: str) -> str:
    """Use this function to place an order and purchase products."""
    return f"Your {query} order has been successfully placed and will be delivered to your doorstep in 45 minutes."


def get_llm_agent():
    retriever = get_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "search_catalog",
        "Searches and returns information about products sold on noon.com. It will return product details. Query it when you need information about products.",
    )
    tools = []
    tools.append(retriever_tool)
    tools.append(buy_product)

    llm = get_llm()
    agent = create_openai_tools_agent(
        llm, tools, hub.pull("hwchase17/openai-tools-agent")
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history


def initialize_session_state():
    st.set_page_config(page_title="Noon Chatbot", page_icon="🟡", layout="wide")
    st.title(":orange[Noon] Chatbot")
    st.header("", divider="rainbow")
    st.sidebar.title("About")
    st.sidebar.info(
        "This chatbot uses GPT 3.5 Turbo with all-mpnet-base-v2 embeddings."
    )
    if len(history.messages) == 0:
        history.add_ai_message("Hi there! Welcome to noon. How can I help you?")
    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = get_llm_agent()


def get_llm_agent_from_session() -> LLMChain:
    return st.session_state["llm_chain"]


initialize_session_state()

if len(history.messages) == 0 or st.sidebar.button("Clear message history"):
    history.clear()
    history.add_ai_message("Hi there! Welcome to noon. How can I help you?")

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input("Your message"):
    st.chat_message(USER).write(prompt)
    with st.spinner("Thinking..."):
        stream_handler = StreamHandler(st.empty())
        agent = get_llm_agent_from_session()
        result = agent.invoke(
            {"input": prompt},
            config={
                "callbacks": [stream_handler],
                "configurable": {"session_id": "<foo>"},
            },
        )
        response = result["output"]