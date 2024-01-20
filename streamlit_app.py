from dataclasses import dataclass
from enum import Enum

import streamlit as st

# from dotenv import load_dotenv
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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_openai import ChatOpenAI

history = StreamlitChatMessageHistory(key="st_history_key")
# load_dotenv()

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"


class ChainMethod(Enum):
    STUFF = "stuff"
    MAPREDUCE = "map_reduce"
    REFINE = "refine"


@dataclass
class Message:
    actor: str
    payload: str


@st.cache_resource
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        temperature=0.7,
        model="gpt-4",
        streaming=True,
        verbose=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )


@st.cache_resource
def get_retriever():
    vectorstore = Pinecone.from_existing_index(
        "noon-catalog",
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        "text",
    )
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
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
        template="""You are a helpful assistant for an e-commerce website. You return product catalog and information based on the following pieces of context and chat history to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
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
        llm=get_llm(),
        retriever=get_retriever(),
        verbose=True,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_prompt},
        chain_type=ChainMethod.STUFF.value,
        rephrase_question=False,
        condense_question_prompt=condense_question_prompt,
        condense_question_llm=get_llm(),
        return_source_documents=True,
    )

    return conversation


def initialize_session_state():
    st.title("🦜🔗 Noon Chatbot")
    if len(history.messages) == 0:
        history.add_ai_message("Hi there! How can I help you?")

    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = get_llm_chain_w_customsearch()


def get_llm_chain_from_session() -> LLMChain:
    return st.session_state["llm_chain"]


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
