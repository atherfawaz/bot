import streamlit as st

# from devtools import debug
# from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.memory import StreamlitChatMessageHistory
from langchain.tools.retriever import create_retriever_tool

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# load_dotenv()

USER = "user"
ASSISTANT = "ai"
history = StreamlitChatMessageHistory()


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


@st.cache_resource
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        temperature=0,
        model="gpt-4-1106-preview",
        streaming=True,
        verbose=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )


@st.cache_resource
def get_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(
        "catalog-1536",
        embeddings,
        "text",
    )
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    return retriever


def get_llm_agent():
    retriever = get_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "search_electronics_and_home_appliances",
        "Searches and returns information about products sold on noon.com. Query it when you need information about electronics and home appliances.",
    )
    tools = []
    tools.append(retriever_tool)

    llm = get_llm()
    agent_prompt: ChatPromptTemplate = hub.pull("hwchase17/openai-tools-agent")
    agent_prompt.messages[0] = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            template="""
            You are an ecommerce assistant of noon.com.
            Your context is limited to products available on noon.com.
            Only answer questions related to products from electronics and home appliances.
            Prices are provided in the text for the products you receive, so find and return them from there.
            If you find product URLs use them to direct customer to that page.
            If you find image URLs and render images in markdown.
            When asked to compare products, compare them in a tabular format.
            When asked about delivery estimate or order status, direct to customer support.
            When asked about amazon or other websites, say that you are not aware of it.
            """,
        ),
    )
    agent = create_openai_tools_agent(llm, tools, agent_prompt)
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
    st.sidebar.info("This chatbot uses GPT-4 with all-mpnet-base-v2 embeddings.")
    if len(history.messages) == 0:
        history.add_ai_message("Hi there! Welcome to noon. How can I help you?")
    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = get_llm_agent()


def get_llm_agent_from_session() -> LLMChain:
    return st.session_state["llm_chain"]


initialize_session_state()

if len(history.messages) == 0:
    history.clear()
    history.add_ai_message("Hi there! Welcome to noon. How can I help you?")

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

prompt: str = st.chat_input("Ask a question")
if prompt:
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
