import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()

openai_api_key = os.getenv("API_KEY")

# -----------------------------
# Streamlit Page
# -----------------------------
st.set_page_config(
    page_title="Eleven Madison Park RAG Chatbot",
    page_icon="🍽️"
)

st.title("🍽️ Eleven Madison Park RAG Chatbot")
st.write(
    "Ask questions about the restaurant using Retrieval-Augmented Generation (RAG)."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📚 About This Chatbot")

st.sidebar.write("""
### RAG Q&A Chatbot

This application uses:

- LangChain
- OpenAI Embeddings
- ChromaDB
- GPT-4o-mini

Ask questions about Eleven Madison Park.
""")

st.sidebar.subheader("Sample Questions")

st.sidebar.write("""
• Who is Daniel Humm?

• What type of restaurant is Eleven Madison Park?

• What is the restaurant philosophy?

• Where is the restaurant located?
""")

# -----------------------------
# Load Document
# -----------------------------
DATA_FILE_PATH = "eleven_madison_park_data.txt"

with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Split every webpage into one document
sections = text.split("---END OF SOURCE---")

documents = []

for section in sections:
    section = section.strip()
    if section:
        documents.append(Document(page_content=section))

# -----------------------------
# Create Embeddings
# -----------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_api_key
)

# -----------------------------
# Create Vector Database
# -----------------------------
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

# -----------------------------
# Create Retriever
# -----------------------------
retriever = vector_store.as_retriever(
    search_kwargs={"k":4}
)

# -----------------------------
# Create LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0
)

# -----------------------------
# Build RAG Chain
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# -----------------------------
# Chat Interface
# -----------------------------
question = st.text_input(
    "Ask a question about the restaurant"
)

if question:

    with st.spinner("Searching documents and generating answer..."):

        answer = qa_chain.invoke(question)

    st.subheader("Answer")
    st.success(answer["result"])

    st.caption(
        f"Source: {len(answer['source_documents'])} document chunk(s) retrieved."
    )

    st.divider()

    with st.expander("📄 View Retrieved Context"):

        for i, doc in enumerate(answer["source_documents"]):
            st.markdown(f"### Chunk {i+1}")
            st.write(doc.page_content)
            st.divider()

st.divider()

st.caption(
    "Built using LangChain • OpenAI Embeddings • ChromaDB • GPT-4o-mini • Streamlit"
)