from enum import Enum

import streamlit as st

# from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain,
)
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.memory import (
    ConversationBufferMemory,
    StreamlitChatMessageHistory,
)
from langchain.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings

history = StreamlitChatMessageHistory()
# load_dotenv()

USER = "user"
ASSISTANT = "ai"


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


class ChainMethod(Enum):
    STUFF = "stuff"
    MAPREDUCE = "map_reduce"
    REFINE = "refine"


@st.cache_resource
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        verbose=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )


@st.cache_resource
def get_retriever():
    llm = OpenAI(temperature=0)
    metadata_field_info = [
        AttributeInfo(
            name="image_url",
            description="The link or URL of the image of the product",
            type="string",
        ),
        AttributeInfo(
            name="price",
            description="The price of the product",
            type="string",
        ),
        AttributeInfo(
            name="product_url",
            description="The link or URL of the product",
            type="string",
        ),
        AttributeInfo(
            name="rating",
            description="The rating of the product",
            type="number",
        ),
        AttributeInfo(
            name="sku",
            description="The SKU or ID of the product",
            type="string",
        ),
        AttributeInfo(
            name="title",
            description="The title or the name of the product",
            type="string",
        ),
    ]
    vectorstore = Pinecone.from_existing_index("catalog-v2", OpenAIEmbeddings(), "text")
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        "Product details and specifications",
        metadata_field_info,
        verbose=True,
    )
    return retriever


def get_llm_chain_w_customsearch():
    combine_prompt = PromptTemplate(
        template="""
        You are a helpful and friendly ecommerce assistant, your context is limited to the data passed to you and nothing else.
        You deal with electronics and home appliances categories only, for other categories and their products, direct to noon: https://www.noon.com/uae-en/.
        Price, product_url, image_url for a product is provided in the text for the products you receive, so find them from there.
        When given a price range in the search query, only show products that meet the criteria. If nothing meets it, say you don't have the products.
        When asked about amazon or other websites, say that you are not aware of it.
        Render all links and images in markdown format.
        Also give the specifications, SKU, rating, and warranty with each product.
        Present comparisons and multiple products in a tabular format.
        For any other questions, problems, or complaints, direct to customer support here: https://help.noon.com/hc/en-us.

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
    st.set_page_config(page_title="Noon Chatbot", page_icon="🟡", layout="wide")
    st.title(":orange[Noon] Chatbot")
    st.header("", divider="rainbow")
    st.sidebar.title("About")
    st.sidebar.info("This chatbot uses GPT 3.5 Turbo with OpenAI embeddings.")
    if len(history.messages) == 0:
        history.add_ai_message("Hi there! Welcome to noon. How can I help you?")
    if "llm_chain" not in st.session_state:
        st.session_state["llm_chain"] = get_llm_chain_w_customsearch()


def get_llm_chain_from_session() -> LLMChain:
    return st.session_state["llm_chain"]


initialize_session_state()

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input("Ask a question"):
    prompt = prompt.strip()
    st.chat_message(USER).write(prompt)
    with st.spinner("Please wait.."):
        stream_handler = StreamHandler(st.empty())
        llm_chain = get_llm_chain_from_session()
        response = llm_chain.invoke(
            {"question": prompt, "chat_history": history.messages, "input": prompt},
            config={
                "callbacks": [stream_handler],
                "configurable": {"session_id": "<foo>"},
            },
        )
        st.chat_message(ASSISTANT).write(response["answer"])
