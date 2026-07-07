# Production-Ready AI PDF RAG Chatbot

A portfolio-quality, production-ready **Retrieval-Augmented Generation (RAG) Chatbot** built using Python 3.14, Streamlit, ChromaDB, and the latest modular LangChain v0.3+ architecture. 

This application allows users to upload multiple PDF documents and engage in a context-aware conversation with those documents, complete with source citations (filenames and page numbers).

---

## 🏗️ System Architecture

The chatbot utilizes a standard two-phase RAG lifecycle:

```
[Ingestion Phase]
PDF Files ──> PyPDF Parser ──> Document Objects (with Page Metadata)
                  │
                  ▼
       Recursive Character Splitter (1000 Char / 200 Overlap)
                  │
                  ▼
       OpenAI text-embedding-3-small (1536 Dimensions)
                  │
                  ▼
       ChromaDB (In-Memory Vector Indexing)

[Retrieval & Generation Phase]
User Question ──> LLM Standalone Reformulation (Contextualization)
                          │
                          ▼
                  Similarity Search (ChromaDB) ──> Relevant Context Chunks
                                                          │
                                                          ▼
                                            GPT-4o-mini Generator ──> Answer + Citations
```

### Key Technical Patterns:
* **History-Aware Condensation**: Prior to querying the database, the chatbot passes the message history and the user's input to a condensation chain to formulate a standalone search query. This resolves pronoun ambiguity (e.g. "it", "they") in follow-up queries.
* **Metadata Propagation**: Document page numbers and file names are preserved during splitting, allowing the app to output precise page-level citations for every response.
* **Performance Caching**: Employs Streamlit's `st.session_state` to cache the vector database, eliminating redundant Embeddings API calls on user reruns.

---

## 🛠️ Tech Stack & Dependencies

* **Python 3.14**
* **Streamlit**: Conversational UI and layout framework.
* **LangChain (v0.3+)**: Modular LLM abstraction framework.
* **ChromaDB**: High-performance in-memory vector storage.
* **OpenAI API**: 
  * `text-embedding-3-small` (for semantic vectors).
  * `gpt-4o-mini` (for response synthesis).
* **PyPDF**: PDF text extraction.
* **python-dotenv**: Environment configuration.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python installed (Python 3.10 to 3.14).

### 2. Clone and Setup
Extract or clone the project folder, then navigate into the directory:
```bash
cd AI-PDF-RAG-Chatbot
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your OpenAI API Key:
```bash
cp .env.example .env
```
Open `.env` and configure:
```env
OPENAI_API_KEY=your-api-key-here
```

### 5. Run the Chatbot
Start the Streamlit application:
```bash
streamlit run app.py
```
A browser tab will open automatically at `http://localhost:8501`.

---

## 🛡️ Best Practices & Quality Standards
* **Security**: API keys are loaded via environment variables and never printed to the logs.
* **PEP8 Compliance**: Standard indentation, modular separation of concerns (`utils.py` contains business logic; `app.py` handles presentation).
* **Robust Error Handling**: Standard try-except blocks prevent app crashes on faulty PDFs or API rate limits.
