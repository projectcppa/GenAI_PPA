import os
import streamlit as st
import torch
from streamlit_chat import message
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf2image import convert_from_path
import pytesseract
from langchain.docstore.document import Document

# ------------------------
# API Key (Google Gemini)
# ------------------------
GOOGLE_API_KEY = "AIzaSyB4ixfEQ-4Iv2hfUjM_3GOm6-plaDtnzSE"

# ------------------------
# Page setup
# ------------------------
st.set_page_config(page_title="CPPA Document Chatbot", layout="wide")
st.title("📚 CPPA Document Chatbot")

# ------------------------
# Session state
# ------------------------
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "sources" not in st.session_state:
    st.session_state["sources"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "docs_processed" not in st.session_state:  
    st.session_state["docs_processed"] = False

# ------------------------
# OCR for scanned PDFs
# ------------------------
def ocr_pdf(file_path):
    pages = convert_from_path(file_path)
    docs = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page)
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": os.path.basename(file_path), "page": i + 1},
            ))
    return docs

# ------------------------
# Cache embeddings
# ------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        #model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

# ------------------------
# Upload + Process Documents
# ------------------------
uploaded_files = st.file_uploader(
    "Upload one or more documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

enable_ocr = st.checkbox("Enable OCR for scanned PDFs (may take longer)", value=False)

if uploaded_files and not st.session_state.docs_processed:
    all_new_docs = []
    os.makedirs("temp", exist_ok=True)

    progress_bar = st.progress(0, text="Processing documents...")
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        docs = []
        if file_extension == ".pdf":
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                if not docs and enable_ocr:
                    st.warning(f"Could not extract text from {uploaded_file.name}. Attempting OCR...")
                    docs = ocr_pdf(file_path)
            except Exception as e:
                st.error(f"Error loading PDF {uploaded_file.name}: {e}")
                if enable_ocr:
                    st.warning(f"Attempting OCR for {uploaded_file.name}...")
                    docs = ocr_pdf(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
            docs = loader.load()
        else:
            st.error(f"Unsupported file type: {file_extension}")
            continue

        all_new_docs.extend(docs)

        progress = idx / total_files
        progress_bar.progress(progress, text=f"Processing {uploaded_file.name} ({int(progress*100)}%)")

    progress_bar.empty()

    # Build index ONCE
    with st.spinner("Splitting and indexing documents..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
        chunks = text_splitter.split_documents(all_new_docs)

        embeddings = get_embeddings()
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

    st.session_state.docs_processed = True  #  Mark docs as processed
    st.success(f"{len(uploaded_files)} document(s) added and indexed!")

# ------------------------
# Chat with documents
# ------------------------
if st.session_state.vectorstore:
    query = st.text_input("Ask a question about your documents:")

    if query:
        docs = st.session_state.vectorstore.similarity_search(query, k=4) 

        context_parts, sources = [], []
        seen = set()
        for doc in docs:
            filename = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "?")
            src = f"{filename} (Page {page})"
            if src not in seen:
                context_parts.append(f"[{src}]\n{doc.page_content}")
                sources.append(src)
                seen.add(src)

        context = "\n\n".join(context_parts)

        chat = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
        )

        prompt = f"""
        You are an expert in analyzing multiple documents.
        Use ONLY the content provided below to answer.
        If the same answer appears in multiple documents, show it separately for each doc with its source.
        If the query asks for a comparison, highlight differences across documents.
        If no document contains the answer, reply:
        "Apologies! I could not find this information in the documents."
        Do not invent or generalize. Be accurate.

        Context:
        {context}

        Question:
        {query}
        """
        
        with st.spinner("Generating response..."):
            response = chat.invoke(prompt).content

        st.session_state.past.append(query)
        st.session_state.generated.append(response)
        st.session_state.sources.append(sources)

# ------------------------
# Display chat
# ------------------------
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(
            st.session_state["generated"][i] + "\n\nSources: " + ", ".join(st.session_state["sources"][i]),
            key=str(i),
        )
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
