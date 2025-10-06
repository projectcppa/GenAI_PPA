import os
import glob
import torch
import pytesseract
import re
import json
from PIL import Image
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import PDF_DIR, CHROMA_DIR, ABBREV_FILE, EMBEDDINGS_MODEL, DEVICE

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def clean_chroma_dir(chroma_dir):
    """Safely remove Chroma DB files without deleting the folder."""
    if not os.path.exists(chroma_dir):
        return
    # Only remove DB and metadata files
    for pattern in ["*.sqlite3", "*.bin", "*.pkl", "*.json"]:
        for file in glob.glob(os.path.join(chroma_dir, pattern)):
            try:
                os.remove(file)
            except PermissionError:
                print(f"⚠️ Cannot remove {file} — file is in use.")
    # Ensure folder exists
    os.makedirs(chroma_dir, exist_ok=True)


def run_ingestion():
    # Step 1: Reset Chroma DB safely
    clean_chroma_dir(CHROMA_DIR)

    # Step 2: Load PDFs with OCR fallback
    all_docs = []
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(PDF_DIR, filename)
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                if not any(p.page_content.strip() for p in pages):
                    images = convert_from_path(file_path, dpi=300)
                    pages = []
                    for i, image in enumerate(images):
                        text = pytesseract.image_to_string(image, lang="eng")
                        text = ' '.join(text.split())
                        if text.strip():  # skip empty OCR
                            pages.append(Document(
                                page_content=text,
                                metadata={"source": filename, "page": i + 1}
                            ))
                else:
                    for i, page in enumerate(pages):
                        page.metadata["source"] = filename
                        page.metadata["page"] = i + 1

                all_docs.extend(pages)

            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

    if not all_docs:
        return {"status": "error", "message": "❌ No documents loaded. Check your PDF folder."}

    # Step 3: Split into chunks
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)

    if not chunks:
        return {"status": "error", "message": "No chunks created from documents."}

    # Step 4: Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        model_kwargs={"device": DEVICE}
    )

    # Step 5: Save Chroma Vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()

    # Step 6: Extract abbreviations
    abbr_dict = {}
    pattern = r"\b[A-Z]{2,}[a-zA-Z0-9]*[a-z]?\b"
    for doc in chunks:
        matches = re.findall(pattern, doc.page_content)
        for m in matches:
            entry = {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "text": doc.page_content[:400]
            }
            abbr_dict.setdefault(m, []).append(entry)

    with open(ABBREV_FILE, "w", encoding="utf-8") as f:
        json.dump(abbr_dict, f, indent=2)

    return {"status": "success", "chunks": len(chunks)}

 