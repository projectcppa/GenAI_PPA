import os
import shutil
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------------
# Configuration
# ------------------------
PDF_DIR = "CPPA PPA PDF"            # 📂 Folder with your PDF files
FAISS_DIR = "faiss_db"      # 📁 Where FAISS index will be saved

# ------------------------
# Step 1: Clean Old Index
# ------------------------
if os.path.exists(FAISS_DIR):
    shutil.rmtree(FAISS_DIR)
    print("🧹 Old FAISS index removed.")

# ------------------------
# Step 2: Load PDFs with Metadata
# ------------------------
all_docs = []

for filename in os.listdir(PDF_DIR):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(PDF_DIR, filename)
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            for i, page in enumerate(pages):
                # Attach filename + 1-based page number
                page.metadata["source"] = filename
                page.metadata["page"] = i + 1
                all_docs.append(page)

            print(f"✅ Loaded: {filename} ({len(pages)} pages)")

        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

if not all_docs:
    raise ValueError("❌ No documents loaded. Check your PDF folder.")

# ------------------------
# Step 3: Split into Chunks
# ------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)

print(f"🧩 Total chunks created: {len(chunks)}")

# ------------------------
# Step 4: Generate Embeddings
# ------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# ------------------------
# Step 5: Create and Save FAISS
# ------------------------
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(FAISS_DIR)

print(f"💾 FAISS index saved to: {FAISS_DIR}")
