import os
import shutil
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

# ------------------------
# Configuration
# ------------------------
PDF_DIR = r"C:\Users\Alina.Javed\Documents\PPA AI\GenAI_PPA\1-Alina_Working\Alina_Working\CPPA PPA PDF"
CHROMA_DIR = r"C:\Chroma"
ABBREV_FILE = os.path.join(CHROMA_DIR, "abbreviations.json")

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------------
# Step 1: Clean Old Index
# ------------------------
if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
    print("🧹 Old Chroma index removed.")
os.makedirs(CHROMA_DIR, exist_ok=True)

# ------------------------
# Step 2: Load PDFs with OCR Fallback
# ------------------------
all_docs = []
for filename in os.listdir(PDF_DIR):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(PDF_DIR, filename)
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            if not any(p.page_content.strip() for p in pages):
                print(f"⚠️ {filename} seems scanned → using OCR...")
                pages = []
                images = convert_from_path(file_path, dpi=300)
                for i, image in enumerate(images):
                    text = pytesseract.image_to_string(image, lang="eng")
                    text = ' '.join(text.split())
                    pages.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": i + 1}
                    ))
            else:
                for i, page in enumerate(pages):
                    page.metadata["source"] = filename
                    page.metadata["page"] = i + 1

            all_docs.extend(pages)
            print(f"✅ Loaded: {filename} ({len(pages)} pages)")

        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

if not all_docs:
    raise ValueError("❌ No documents loaded. Check your PDF folder.")

# ------------------------
# Step 3: Split into Chunks
# ------------------------
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(all_docs)
print(f"🧩 Total chunks created: {len(chunks)}")

# ------------------------
# Step 4: Embeddings
# ------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# ------------------------
# Step 5: Create and Save Chroma Vectorstore
# ------------------------
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)
print(f"💾 Chroma index saved to: {CHROMA_DIR}")

# ------------------------
# Step 6: Extract Abbreviations + Their Chunks
# ------------------------
abbr_dict = {}
pattern = r"\b[A-Z]{2,}[a-zA-Z0-9]*[a-z]?\b"

for doc in chunks:
    matches = re.findall(pattern, doc.page_content)
    for m in matches:
        entry = {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "text": doc.page_content[:400]  # preview
        }
        abbr_dict.setdefault(m, []).append(entry)

with open(ABBREV_FILE, "w", encoding="utf-8") as f:
    json.dump(abbr_dict, f, indent=2)

print(f"📑 Abbreviation index with context saved to {ABBREV_FILE}")
