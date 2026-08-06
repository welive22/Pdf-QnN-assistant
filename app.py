"""
app.py - PDF Question Answering App using RAG (Retrieval-Augmented Generation)

Built for Epochs '26 Assignment 11 (final assignment!).

Upload a PDF -> it gets split into chunks and turned into embeddings ->
stored in a Chroma vector database -> when you ask a question it finds the
most relevant chunks and passes them + your question (+ chat history) to
Google Gemini to answer.

Free LLM: Google Gemini API (Free Tier)
"""

import os
import shutil
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Model settings
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

# Temp directory for Chroma vector store
CHROMA_BASE_DIR = tempfile.mkdtemp(prefix="chroma_")


def get_or_create_vectorstore(pdf_file, api_key):
    """Loads the PDF, splits into chunks, embeds them, and stores in Chroma."""
    # Save uploaded file temporarily
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(pdf_file.read())
    temp_pdf.close()

    # Load PDF
    loader = PyPDFLoader(temp_pdf.name)
    documents = loader.load()

    if len(documents) == 0:
        return None, "Couldn't read any text from that PDF (might be a scanned/image PDF)."

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # Generate embeddings using Gemini's own embedding model
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)

    # Store in Chroma (fresh folder per upload)
    session_dir = tempfile.mkdtemp(dir=CHROMA_BASE_DIR)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=session_dir,
    )

    # Clean up temp PDF
    os.unlink(temp_pdf.name)

    return vectorstore, f"PDF processed! Split into {len(chunks)} chunks."


def create_qa_chain(vectorstore, api_key):
    """Builds the RAG chain with conversation memory."""
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0.2,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
    )

    return qa_chain


# ─── Streamlit UI ───────────────────────────────────────────────

st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄", layout="wide")

st.title("📄 PDF Question Answering Assistant")
st.markdown(
    "Upload a PDF, process it, and ask questions about its content. "
    "The assistant remembers your conversation so you can ask follow-up questions naturally."
)

# Sidebar
with st.sidebar:
    st.header("Settings")
    gemini_api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key here",
        help="Get a free key at https://aistudio.google.com/apikey",
    )
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    process_btn = st.button("Process PDF", type="primary", use_container_width=True)

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "status_msg" not in st.session_state:
    st.session_state.status_msg = ""

# Process PDF
if process_btn and uploaded_file and gemini_api_key:
    with st.spinner("Reading PDF and building vector store..."):
        vectorstore, msg = get_or_create_vectorstore(uploaded_file, gemini_api_key)
        if vectorstore is not None:
            st.session_state.vectorstore = vectorstore
            st.session_state.qa_chain = create_qa_chain(vectorstore, gemini_api_key)
            st.session_state.status_msg = msg
            st.session_state.chat_history = []
        else:
            st.session_state.status_msg = msg

if process_btn and uploaded_file and not gemini_api_key:
    st.session_state.status_msg = "Please enter your Gemini API key in the sidebar."

# Main area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Status")
    if st.session_state.status_msg:
        st.info(st.session_state.status_msg)
    else:
        st.text("No PDF loaded yet.")

with col2:
    st.subheader("Chat")
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Question input
    question = st.chat_input("Ask a question about the PDF...")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        if st.session_state.qa_chain is not None:
            # Get answer from RAG chain
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.qa_chain.invoke({"question": question})
                    answer = result["answer"]
                    st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            with st.chat_message("assistant"):
                st.markdown("Please upload and process a PDF first.")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "Please upload and process a PDF first."}
            )

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        if st.session_state.qa_chain is not None:
            st.session_state.qa_chain.memory.clear()
        st.rerun()
