import os
import streamlit as st
from streamlit_chat import message
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS


# ------------------ CONFIG ------------------
st.set_page_config(page_title="Document Chatbot", layout="wide")
st.title("📚 Document Chat with Gemini")

# Gemini API key (set in environment or paste here)
os.environ["GOOGLE_API_KEY"] = "AIzaSyB4ixfEQ-4Iv2hfUjM_3GOm6-plaDtnzSE"

# Embeddings
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        #model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
embeddings = get_embeddings()

# ------------------ FUNCTIONS ------------------
def create_vector_store(file_path, persist_path="faiss_index"):
    """Index the uploaded document and save FAISS locally"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_path)
    return True


def load_vector_store(persist_path="faiss_index"):
    """Load FAISS index from disk"""
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)


def query_doc(question, chat_history, persist_path="faiss_index"):
    """Query LLM with chat history + retrieved context"""
    vectorstore = load_vector_store(persist_path)
    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # Retrieve context
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([d.page_content for d in docs])

    # Convert chat history to text
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])

    final_prompt = f"""
        You are an expert in analyzing multiple documents.

        Use ONLY the content provided below to answer.

        If the same answer appears in multiple documents, show it separately for each doc with its source.

        If the query asks for a comparison, highlight differences across documents.

        If no document contains the answer, reply:

        "Apologies! I could not find this information in the documents."

        Do not invent or generalize. Be accurate. Use the following context and chat history to answer:

Chat History:
{history_text}

Context:
{context}

Question: {question}
"""

    response = llm.invoke(final_prompt)
    answer = response.content

    # Append new interaction
    chat_history.append((question, answer))
    return answer, chat_history


# ------------------ STREAMLIT UI ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "faiss_ready" not in st.session_state:
    st.session_state.faiss_ready = False

# File upload
uploaded_file = st.file_uploader("📂 Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None and not st.session_state.faiss_ready:
    file_path = os.path.join("uploaded_" + uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Indexing document... ⏳"):
        create_vector_store(file_path)
    st.session_state.faiss_ready = True
    st.success("✅ Document indexed successfully!")

# Chat interface
if st.session_state.faiss_ready:
    query = st.text_input("💬 Ask a question about your document:")

    if st.button("Send") and query:
        with st.spinner("Thinking... 🤔"):
            answer, st.session_state.chat_history = query_doc(query, st.session_state.chat_history)

    # Display chat history
    for i, (q, a) in enumerate(st.session_state.chat_history):
        message(q, is_user=True, key=f"user_{i}")
        message(a, is_user=False, key=f"bot_{i}")
