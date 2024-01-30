from dataclasses import dataclass
from enum import Enum

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain,
)
from langchain.memory import (
    ConversationBufferMemory,
    StreamlitChatMessageHistory,
)
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

from ancillaries.perplexity import PerplexityChat

load_dotenv()

history = StreamlitChatMessageHistory()

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"


@dataclass
class Message:
    actor: str
    payload: str


class ChainMethod(Enum):
    STUFF = "stuff"
    MAPREDUCE = "map_reduce"
    REFINE = "refine"


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


def get_llm_for_perplexity():
    return PerplexityChat(model_name="llama-2-70b-chat", temperature=0.0, verbose=True)


@st.cache_resource
def get_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index("catalog-v2", embeddings, "text")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    return retriever


def get_llm_chain_w_customsearch():
    combine_prompt = PromptTemplate(
        template="""
        Your name is Nora, and you are an ecommerce assistant of noon.com.
        Your context is limited to the data passed to you.
        Only answer questions related to products from electronics and home appliances.
        Prices and product links are provided in the text for the products you receive, so find and return them from there.
        If you find product URLs use them to direct customer to that page.
        Do not return image details at all.
        Limit your results to only 4 products at maximum.
        When listing multiple products, write only one line for each product describing its price and specifications.
        Minutes and Rocket are part of noon, so you should answer questions related to it.
        When asked about amazon or other websites, say that you are not aware of it.
        For problems or complaints, direct to customer support.
        You were created and built by noon.com.
        
        Context: {context}
        Chat history: {chat_history}
        Question: {question} 
        Helpful Answer:""",
        input_variables=["context", "question", "chat_history"],
    )

    condense_question_prompt = PromptTemplate.from_template(
        template="""
        Return text in the original language of the follow up question.
        Never rephrase the follow up question given the chat history unless the follow up question needs context.
        
        Chat History: {chat_history}
        Follow Up question: {question}
        Standalone question:
        """
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        human_prefix=USER,
        ai_prefix=ASSISTANT,
        return_messages=True,
        chat_memory=history,
    )

    conversation = ConversationalRetrievalChain.from_llm(
        llm=get_llm_for_perplexity(),
        retriever=get_retriever(),
        verbose=True,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_prompt},
        chain_type=ChainMethod.STUFF.value,
        rephrase_question=False,
        condense_question_prompt=condense_question_prompt,
        condense_question_llm=get_llm_for_perplexity(),
        return_source_documents=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    return conversation


def initialize_session_state():
    st.set_page_config(page_title="Noon Chatbot", page_icon="🟡", layout="wide")
    st.title(":orange[Noon] Chatbot")
    st.header("", divider="rainbow")
    st.sidebar.title("About")
    st.sidebar.info(
        "This chatbot uses Perplexity AI with all-mpnet-base-v2 embeddings."
    )
    if len(history.messages) == 0:
        history.add_ai_message("Hi there! Welcome to noon. How can I help you?")
    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = get_llm_chain_w_customsearch()


def get_llm_chain_from_session() -> LLMChain:
    return st.session_state["llm_chain"]


def get_summarization_from_sessoin() -> str:
    return st.session_state["chat_history_summarization"]


initialize_session_state()

msg: Message
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

prompt: str = st.chat_input("Enter a prompt here")

if prompt:
    st.chat_message(USER).write(prompt)
    with st.spinner("Please wait.."):
        print(f"YOUR PROMPT={prompt}")
        stream_handler = StreamHandler(st.empty())
        llm_chain = get_llm_chain_from_session()
        result = llm_chain.invoke({"input": prompt, "question": prompt})
        st.chat_message(ASSISTANT).write(result["answer"])
