import streamlit as st
from pypdf import PdfReader
#from utils import extract_text_from_pdf
from utils import (
    extract_text_from_pdf,
    chunk_text,
    create_vectorstore,
    answer_question_with_history,
)
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv(override=True)

# Verify API Key exists
if not os.getenv("OPENAI_API_KEY"):
    st.error("🔑 OpenAI API Key is missing! Please configure it in your `.env` file.")
    st.stop()
# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# Inject Custom CSS for Premium UI Aesthetics
st.markdown("""
    <style>
    /* Premium dark aesthetics */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    
    /* Expander customization */
    .streamlit-expanderHeader {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
    }
    
    /* Divider colors */
    hr {
        border-color: #334155 !important;
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0F172A;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------
st.title("📄 AI PDF RAG Chatbot")
st.write(
    "Upload any PDF and ask questions using Retrieval-Augmented Generation (RAG)."
)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    st.divider()

    st.markdown("### 💡 Sample Questions")

    st.markdown("""
- Summarize the document.
- What is the main topic?
- Explain chapter 2.
- List important points.
""")

# ----------------------------
# Main Area
# ----------------------------
if not uploaded_files:
    st.info("👈 Upload one or more PDF files from the sidebar to begin.")
    # Reset session state if files are removed
    if "vectorstore" in st.session_state:
        del st.session_state["vectorstore"]
        del st.session_state["file_names"]
    if "messages" in st.session_state:
        del st.session_state["messages"]

else:
    # Build a clean list of filenames
    uploaded_filenames = [f.name for f in uploaded_files]
    
    # Render successful upload status
    st.success(f"✅ Active documents: {', '.join(uploaded_filenames)}")

    # Build vectorstore only if not cached or if the files set changed
    cached_filenames = st.session_state.get("file_names", [])
    if "vectorstore" not in st.session_state or set(uploaded_filenames) != set(cached_filenames):
        with st.spinner("Analyzing document(s), creating embeddings, and building vector database..."):
            try:
                all_docs = []
                for uploaded_file in uploaded_files:
                    docs = extract_text_from_pdf(uploaded_file)
                    all_docs.extend(docs)
                
                chunks = chunk_text(all_docs)
                vectorstore = create_vectorstore(chunks)
                
                # Cache in session state
                st.session_state["vectorstore"] = vectorstore
                st.session_state["file_names"] = uploaded_filenames
                st.session_state["pages_count"] = len(all_docs)
                st.session_state["total_chars"] = sum(len(doc.page_content) for doc in all_docs)
                st.session_state["chunks_count"] = len(chunks)
                # Clear chat history when files change
                st.session_state["messages"] = []
            except Exception as e:
                st.error(f"❌ Error building vector store: {e}")
                st.stop()

    # Retrieve vectorstore and stats from session state
    vectorstore = st.session_state["vectorstore"]

    # Display document statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pages", st.session_state["pages_count"])
    with col2:
        st.metric("Total Characters", st.session_state["total_chars"])
    with col3:
        st.metric("Number of Chunks", st.session_state["chunks_count"])

    st.divider()

    # Similarity Search Debug Tool (optional dropdown)
    with st.expander("🛠️ Debug Vector Database Search"):
        st.write("Type a query below to retrieve the most semantically relevant chunks from ChromaDB.")
        search_query = st.text_input("Enter search query:", key="debug_search")
        if search_query:
            with st.spinner("Searching ChromaDB..."):
                results = vectorstore.similarity_search_with_score(search_query, k=3)
                st.success(f"Found {len(results)} relevant chunks:")
                for idx, (doc, score) in enumerate(results):
                    st.markdown(f"**Result {idx + 1} | Page {doc.metadata.get('page')} | Source: {doc.metadata.get('source')} | Distance Score: {score:.4f}**")
                    st.write(doc.page_content)
                    st.divider()

    # ----------------------------
    # Conversational Chat Area (Step 4)
    # ----------------------------
    st.subheader("💬 Chat with Documents")
    
    # Initialize message list
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 View Sources"):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(f"**Source: {src['source']} | Page: {src['page']}**")
                        st.write(src["content"])
                        st.divider()

    # Chat input
    question = st.chat_input("Ask a question about your PDF documents...")

    if question:
        # Display user message immediately
        with st.chat_message("user"):
            st.write(question)
        
        # Save user message to session state
        st.session_state["messages"].append({"role": "user", "content": question})

        # Build chat history for LangChain
        chat_history = []
        for msg in st.session_state["messages"][:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, source_docs = answer_question_with_history(
                        vectorstore, question, chat_history
                    )
                    st.write(answer)

                    # Format and save sources
                    sources = []
                    for doc in source_docs:
                        sources.append({
                            "source": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page", "Unknown"),
                            "content": doc.page_content
                        })

                    # Display retrieved sources
                    with st.expander("📚 View Sources"):
                        for idx, src in enumerate(sources):
                            st.markdown(f"**Source: {src['source']} | Page: {src['page']}**")
                            st.write(src["content"])
                            st.divider()

                    # Save assistant message to session state
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"❌ Error generating answer: {e}")
