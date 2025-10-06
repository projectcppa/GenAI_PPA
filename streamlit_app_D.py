import os
import streamlit as st
import torch
torch.cuda.empty_cache()
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
# API Key
# ------------------------
GOOGLE_API_KEY = "AIzaSyB4ixfEQ-4Iv2hfUjM_3GOm6-plaDtnzSE"


# ------------------------
# Page setup
# ------------------------
st.set_page_config(page_title="CPPA Document Chatbot", layout="wide")
st.title("CPPA Document Chatbot")


# ------------------------
# Session state initialization
# ------------------------
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "sources" not in st.session_state:
    st.session_state["sources"] = []

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None


# ------------------------
# Helper: OCR for scanned PDFs
# ------------------------
def ocr_pdf(file_path):
    pages = convert_from_path(file_path)
    docs = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page)
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": os.path.basename(file_path), "page": i + 1},
                )
            )
    return docs


# ------------------------
# Step 1: Upload + Process Multiple Documents
# ------------------------
uploaded_files = st.file_uploader(
    "Upload one or more documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    all_docs = []
    os.makedirs("temp", exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join("temp", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Choose loader
        if uploaded_file.name.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                # If nothing extracted → assume scanned PDF → use OCR
                if not docs or all(not d.page_content.strip() for d in docs):
                    docs = ocr_pdf(file_path)

            except Exception:
                # fallback OCR
                docs = ocr_pdf(file_path)

        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = uploaded_file.name

        else:  # txt
            loader = TextLoader(file_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = uploaded_file.name

        all_docs.extend(docs)  # collect all docs

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(all_docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create a single FAISS vectorstore across all docs
    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success(f"{len(uploaded_files)} document(s) processed and ready for Q&A!")


# ------------------------
# Step 2: Chat with Document (Enter to send, no button)
# ------------------------
if st.session_state.vectorstore:
    query = st.text_input("Ask a question about your documents:")

    if query:  # directly run when user presses Enter
        # Search relevant chunks
        docs = st.session_state.vectorstore.max_marginal_relevance_search(
            query, k=6, fetch_k=20
        )

        # Prepare context with metadata
        context_parts = []
        sources = set()  # deduplicate sources
        for doc in docs:
            filename = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "?")
            src = f"{filename} (Page {page})"

            context_parts.append(f"From {src}:\n{doc.page_content}")
            sources.add(src)

        context = "\n\n".join(context_parts)
        sources = sorted(list(sources))  # clean sorted list

        # Gemini LLM
        chat = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
        )

        # Prompt
        prompt = f"""
        You are an expert in reading and analyzing multiple documents.  
        Use ONLY the content below to answer the question.   

        Rules:
        - If the same answer is found in multiple documents, show each document's answer separately in this format: (DocumentName.pdf, Page N). 
        - Only include the document(s) and page(s) where you actually found the information. 
        - Do not list all documents retrieved, only the one(s) used in your answer.
        - If the answer is not in the documents, reply with:
        "Apologies! I could not find this information in the documents."

        Context:
        {context}

        Question:
        {query}
        """

        response = chat.invoke(prompt).content

        # Save chat history
        st.session_state.past.append(query)
        st.session_state.generated.append(response)
        st.session_state.sources.append(sources)


# ------------------------
# Step 3: Display Chat
# ------------------------
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
         # AI answer
        message(st.session_state["generated"][i], key=str(i))
        # User query
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

      #  message(
       #     st.session_state["generated"][i]
        #    + "\n\nSources: " + ", ".join(st.session_state["sources"][i]),
         #   key=str(i),
        #)
        # User query
        #message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
