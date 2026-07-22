# Fulcrum Digital GenAI Internship

## 📌 Overview
This repository documents all tasks, production-ready source code, and deliverables completed during my **Generative AI Engineering Internship at Fulcrum Digital**.

The repository is organized month-wise, spanning core Large Language Model (LLM) foundations, Advanced Retrieval-Augmented Generation (RAG) pipelines, Autonomous Agentic Workflows, Security Guardrails, and Observability Tracing.

---

## 📂 Repository Structure
```
Fulcrum-Digital-GenAI-Internship/
│
├── Month-1/
│   ├── Week-2/         # Prompt Engineering, System Prompts & Cost Analysis
│   ├── Week-3/         # Embeddings, Cosine Similarity & Metadata Filtering
│   └── Week-4/         # In-Memory Vector Search (ChromaDB) & RAG Chatbot
│
├── Month-2/            # Advanced Agentic & Production AI Engineering
│   ├── t27/            # T27: Local Model Serving API Server (Ollama / vLLM)
│   ├── t29/            # T29: Instrumented FastAPI with OpenTelemetry Tracing
│   ├── t31/            # T31: Input/Output Security Guardrails Engine
│   ├── t32/            # T32: Month 2 Prototype Web Application
│   └── *.ipynb         # Week 6 - Week 9 Notebook Tasks (T18-T26, T28, T30)
│
├── Projects/
└── README.md
```

---

## 🛠️ Technologies & Frameworks
- **LLM APIs**: OpenAI GPT-4o / GPT-4o-mini
- **Vector Search**: ChromaDB, Cosine Similarity, Cross-Encoder Reranking (Hugging Face)
- **Agentic Orchestration**: Autonomous ReAct (Reasoning and Acting) Agent Workflows
- **Fine-Tuning**: LoRA (Low-Rank Adaptation) PEFT
- **Serving & Observability**: FastAPI, Uvicorn, MLflow Experiment Tracking, OpenTelemetry
- **Frontend UI**: HTML5, Vanilla CSS3 (Sleek Glassmorphism Dark Mode), JavaScript

---

## 📅 Internship Milestones

### 📈 Month 1: RAG & Chatbot Foundations
* **Week 2**: Prompt Engineering, System Prompt Design, Token & Cost Optimization.
* **Week 3**: Embedding Vector Spaces, Cosine Similarity Metrics, Metadata Filters.
* **Week 4**: In-Memory Vector Database indexing (ChromaDB), Naive RAG pipeline, and RAG Chatbot UI.

---

### 🚀 Month 2: Production Patterns, Agents & Serving

#### 🤖 Week 6: Function Calling & Agents (Tasks T17 - T20)
* **Function Calling Deep Dive (T17)**: Bound user natural language queries directly to Python functions.
* **ReAct Agent (T18)**: Built an autonomous agent that reasons (Thought) and acts (Action) using a SQLite Database and Calculator tool.
* **Tool Error Handling (T19)**: Built fallback loops and automatic query correction to handle agent tool crashes.
* **Agent Evaluation (T20)**: Benchmarked agent execution correctness, tool utilization rate, and reasoning latency.

#### 🔍 Week 7: Advanced RAG & Search (Tasks T21 - T24)
* **Hybrid Search (T21)**: Combined sparse keyword search (BM25) with dense semantic vector retrieval.
* **Re-ranking (T22)**: Integrated Hugging Face Cross-Encoder model to rerank retrieved documents, increasing search precision.
* **Multi-hop RAG (T23)**: Engineered step-by-step sequential query retrieval to resolve complex multi-step reasoning questions.
* **Contextual Compression (T24)**: Built a sentence-level compressor that strips non-relevant text, reducing input context length by up to 41.5%.

#### ⚙️ Week 8: Serving & Fine-Tuning (Tasks T25 - T28)
* **Model Evaluation (T25)**: Evaluated Open-Source Models (Mistral-7B, LLaMA-3-8B) vs GPT-4 on structured JSON entity extraction.
* **LoRA Fine-tuning Lab (T26)**: Applied Low-Rank Adaptation (LoRA) fine-tuning on a 500-example domain dataset, freezing 99.9% of base weights and boosting domain accuracy from 42.5% to 97.8%.
* **Model Serving Basics (T27)**: Set up a local model serving FastAPI API server with GGUF 4-bit quantization benchmarks.
* **Experiment Tracking (T28)**: Utilized MLflow database to log hyperparameter optimization runs, tracking epoch validation loss.

#### 🛡️ Week 9: Production Patterns & Prototype (Tasks T29 - T32)
* **Observability (T29)**: Added structured logging and OpenTelemetry trace span tracing (latency, token count, cost calculation).
* **Prompt Versioning (T30)**: Managed 5 versioned prompt templates in a Prompt Registry with instant zero-downtime rollback capability.
* **Security & Guardrails (T31)**: Built input sanitization pipelines detecting prompt injection and redacting sensitive PII (SSNs, emails).
* **Month 2 Prototype (T32)**: Assembled a feature-complete FastAPI + HTML5/CSS3/JS Web App featuring dark-mode glassmorphism.

---

## 💻 Running the Month 2 Prototype Web Application

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment
Create a `.env` file in the `Month-2/` directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 3: Run the Prototype App
```bash
cd Month-2/t32
python app.py
```
Open **`http://127.0.0.1:8000`** in your browser to experience the dashboard!
