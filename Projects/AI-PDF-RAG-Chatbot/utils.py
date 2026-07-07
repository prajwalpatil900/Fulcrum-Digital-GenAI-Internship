from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def extract_text_from_pdf(uploaded_file) -> list[Document]:
    """
    Extract text from an uploaded PDF file and return a list of LangChain Document objects.

    Each page in the PDF is represented as a Document with page_content and metadata
    containing the file name (source) and page number (1-indexed).

    Args:
        uploaded_file: The uploaded PDF file-like object from Streamlit.

    Returns:
        list[Document]: A list of Document objects with text and page metadata.
    """
    pdf = PdfReader(uploaded_file)
    documents = []

    for i, page in enumerate(pdf.pages):
        page_text = page.extract_text()
        if page_text:
            metadata = {
                "source": uploaded_file.name,
                "page": i + 1  # Pages are 1-indexed in human terms
            }
            documents.append(Document(page_content=page_text, metadata=metadata))

    return documents

def chunk_text(documents: list[Document]) -> list[Document]:
    """
    Split LangChain Document objects into smaller semantically cohesive chunks.

    Uses RecursiveCharacterTextSplitter to split text into chunks while preserving metadata.

    Args:
        documents (list[Document]): The input list of Documents.

    Returns:
        list[Document]: A list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks: list[Document]) -> Chroma:
    """
    Create a Chroma vector store from chunked Documents using OpenAIEmbeddings.

    Args:
        chunks (list[Document]): List of chunked Document objects.

    Returns:
        Chroma: The Chroma vector database object.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore

def answer_question(vectorstore, question: str) -> tuple[str, list[Document]]:
    """
    Query the vector store for relevant chunks and generate an answer using GPT-4o-mini.

    Args:
        vectorstore (Chroma): The Chroma vector database.
        question (str): The user's query.

    Returns:
        tuple[str, list[Document]]: The generated answer and the list of retrieved source documents.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 1. Retrieve relevant documents
    docs = retriever.invoke(question)

    # 2. Format context
    context = "\n\n".join(
        f"--- Chunk {i+1} (Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}) ---\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    # 3. Create Prompt with guardrails
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional assistant designed to answer questions based on the provided PDF context. "
                   "Strictly use the provided context to answer the question. If the answer cannot be found in the context, "
                   "politely state that you do not know. Do not make up or extrapolate information.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])

    # 4. Setup LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 5. Build LCEL Chain
    chain = prompt | llm | StrOutputParser()

    # 6. Execute chain
    answer = chain.invoke({"context": context, "question": question})

    return answer, docs

def answer_question_with_history(
    vectorstore: Chroma,
    question: str,
    chat_history: list
) -> tuple[str, list[Document]]:
    """
    Execute a conversational RAG pipeline using LangChain.

    1. Re-phrases follow-up questions to standalone queries using the chat history.
    2. Retrieves relevant chunks from ChromaDB using the standalone query.
    3. Synthesizes a factual answer using GPT-4o-mini, considering history and context.

    Args:
        vectorstore (Chroma): The Chroma vector database.
        question (str): The new user question.
        chat_history (list): A list of LangChain BaseMessage objects (HumanMessage/AIMessage).

    Returns:
        tuple[str, list[Document]]: A tuple containing the answer string and the retrieved source documents.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 1. Condensation Step (Standalone Question Generator)
    if len(chat_history) > 0:
        condense_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        condense_prompt = ChatPromptTemplate.from_messages([
            ("system", condense_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        condense_chain = condense_prompt | llm | StrOutputParser()
        standalone_question = condense_chain.invoke({
            "chat_history": chat_history,
            "question": question
        })
    else:
        standalone_question = question

    # 2. Retrieve relevant documents using the standalone query
    docs = retriever.invoke(standalone_question)

    # 3. Format context
    context = "\n\n".join(
        f"--- Chunk {i+1} (Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}) ---\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    # 4. Generate Answer based on retrieved context and history
    qa_system_prompt = (
        "You are a professional assistant designed to answer questions based on the provided PDF context. "
        "Strictly use the provided context to answer the question. If the answer cannot be found in the context, "
        "politely state that you do not know. Do not make up or extrapolate information.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    qa_chain = qa_prompt | llm | StrOutputParser()

    answer = qa_chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question
    })

    return answer, docs