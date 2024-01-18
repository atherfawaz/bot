import streamlit as st
from langchain.agents import (
    AgentType,
    initialize_agent,
)
from langchain.callbacks import StreamlitCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_openai import OpenAI, OpenAIEmbeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
docs = text_splitter.split_documents(documents=[Document(page_content="Ather")])
vectorstore = FAISS.from_documents(
    docs[:50],
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
tool = create_retriever_tool(
    retriever,
    "search_catalog",
    "Searches and returns information about products sold on noon.com. It will return product details. Query it when you need information about products.",
)
tools = [tool]

st_callback = StreamlitCallbackHandler(st.container())

llm = OpenAI(temperature=0.7, streaming=True)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
