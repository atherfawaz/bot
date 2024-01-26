from dataclasses import dataclass
from enum import Enum

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain,
)
from langchain.memory import (
    ConversationBufferMemory,
    # ConversationBufferWindowMemory,
    StreamlitChatMessageHistory,
)
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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


def get_llm_for_perplexity():
    return PerplexityChat(
        model_name="pplx-70b-online",
        temperature=0.7,
        verbose=True,
        # streaming=True,
        # callbacks=[StreamingStdOutCallbackHandler()],
    )


@st.cache_resource
def get_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(
        "catalog-1536",
        embeddings,
        "text",
    )
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    return retriever


def get_llm_chain_w_customsearch():
    condense_question_template = """
    Return text in the original language of the follow up question.
    Never rephrase the follow up question given the chat history unless the follow up question needs context.

    Chat History: {chat_history}
    Follow Up question: {question}
    Standalone question:
    """
    condense_question_prompt = PromptTemplate.from_template(
        template=condense_question_template
    )

    combine_prompt = PromptTemplate(
        template="""
        You are an ecommerce assistant of noon.com.
        Your context is limited to products available on noon.com.
        Prices are provided in the text for the products you receive, so find them from there.
        Always return product URLs and link customers to the product page.
        Always return image URLs and render images as markdown.
        Present multiple products in a tabular format.
        When given a price range in the search query, only show products that meet the criteria. If nothing meets it, say you don't have the products.
        When asked about delivery estimate or order status, direct to customer support.
        When asked about amazon or other websites, say that you are not aware of it.
        
        Context: {context}
        Chat history: {chat_history}
        Question: {question} 
        Helpful Answer:""",
        input_variables=["context", "question", "chat_history"],
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
    # history.add_user_message(prompt)
    st.chat_message(USER).write(prompt)
    with st.spinner("Please wait.."):
        print(f"YOUR PROMPT={prompt}")
        llm_chain = get_llm_chain_from_session()
        response: str = llm_chain(
            {"question": prompt, "chat_history": history.messages}
        )["answer"]
        # history.add_ai_message(response)
        st.chat_message(ASSISTANT).write(response)
