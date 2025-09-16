# PPA.py
# Cleaned + hardened PPA RAG script for VS Code / Python
# Make sure: token.env exists with GEMINI_API_KEY=..., and Tesseract is installed.

import os
import hashlib
import re
import io
from pathlib import Path
from dotenv import load_dotenv

# Load env
load_dotenv("token.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Users\mahnoor.shakir\tesseract.exe")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Put it in token.env as GEMINI_API_KEY=...")

# 3rd-party imports (ensure these packages are installed)
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# Point to Tesseract executable (fallback to env var above)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ------------------ Config ------------------
PDF_PATH = r"C:\Users\mahnoor.shakir\Downloads\karot.pdf"   # update if needed
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------ Helpers ------------------
def sha1_of_file(path: str, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in s)

def clean_text(txt: str) -> str:
    """Remove headers/footers & common PPA boilerplate + normalize whitespace."""
    lines = txt.splitlines()
    drop_patterns = [
        r"^Page\s*\d+", r"^Power Purchase Agreement", r"^Confidential",
        r"^Draft", r"^Executed Version", r"^This Power Purchase Agreement",
        r"^Table of Contents", r"^\s*PPA\s*$", r"^Downloaded on .*", r"^©\s*\d{4}"
    ]
    drop_re = re.compile("|".join(drop_patterns), re.IGNORECASE)
    kept = [ln for ln in lines if ln.strip() and not drop_re.search(ln)]
    text = re.sub(r"\s+\n", "\n", "\n".join(kept))
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def _ocr_pdf_pages(path: str, lang: str = "eng"):
    """Render each page to image and OCR it."""
    pdf = fitz.open(path)
    ocr_docs = []
    for i, page in enumerate(pdf):
        pix = page.get_pixmap(dpi=200)  # increase dpi for better OCR if needed
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang=lang)
        ocr_docs.append(Document(page_content=text, metadata={"source": path, "page": i+1}))
    return ocr_docs

def load_pdf_with_ocr(path: str, min_total_chars: int = 200):
    """
    Try normal text extraction; fallback to OCR if:
      - all pages empty OR
      - total extracted text is below min_total_chars (likely scanned or poorly extracted)
    """
    loader = PyPDFLoader(path)
    docs = loader.load()  # list of Documents
    # Normalize metadata page as int if present
    for d in docs:
        if "page" in d.metadata:
            try:
                d.metadata["page"] = int(d.metadata["page"])
            except Exception:
                pass

    total_len = sum(len(d.page_content.strip()) for d in docs)
    if total_len < min_total_chars:
        print("⚠️ Extracted text is small/empty or scanned PDF — running OCR (this can take a while)...")
        ocr_docs = _ocr_pdf_pages(path, lang="eng")
        # fallback: if OCR produced no text, return original docs (so pipeline doesn't fail)
        if sum(len(d.page_content.strip()) for d in ocr_docs) < 10:
            print("⚠️ OCR produced little or no text — returning original extraction results.")
            return docs
        return ocr_docs
    return docs

# ------------------ Sanity checks ------------------
if not Path(PDF_PATH).exists():
    raise FileNotFoundError(f"PDF not found at {PDF_PATH!r}. Please check the path.")

# Check Tesseract executable exists
if not Path(pytesseract.pytesseract.tesseract_cmd).exists():
    print("Warning: Tesseract executable not found at:", pytesseract.pytesseract.tesseract_cmd)
    print("If OCR is needed, either install Tesseract or set TESSERACT_CMD in token.env to the correct path.")
    # Do not raise here — script can still work if PDF text is extractable

# ------------------ Unique Persist Path ------------------
file_hash = sha1_of_file(PDF_PATH)
file_tag  = sanitize(Path(PDF_PATH).stem.lower())
embed_tag = sanitize(EMBED_MODEL.replace("/", "_"))
PERSIST_DIR = f"./chroma_store/{file_tag}/{embed_tag}/{file_hash[:12]}"
COLLECTION_NAME = f"{file_tag}_{embed_tag}"

# ------------------ Load, Clean & Chunk ------------------
docs = load_pdf_with_ocr(PDF_PATH)

# Clean page text + ensure integer page metadata
for idx, d in enumerate(docs):
    d.page_content = clean_text(d.page_content)
    if "page" not in d.metadata:
        d.metadata["page"] = idx + 1
    else:
        try:
            d.metadata["page"] = int(d.metadata["page"])
        except Exception:
            d.metadata["page"] = idx + 1

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=120,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"Loaded {len(chunks)} cleaned chunks from {PDF_PATH}")

# If no chunks found, bail with advice
if len(chunks) == 0:
    raise RuntimeError("No text chunks were produced. Check that the PDF contains selectable text or OCR is installed and working.")

# ------------------ Embeddings & Vector DB ------------------
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
)
vectordb.persist()

# ------------------ Gemini ------------------
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ------------------ Retrieval Utilities ------------------
def _merge_results_with_scores(*lists, max_items=12):
    seen, merged = set(), []
    def sig(doc):
        head = (doc.page_content or "")[:120].strip()
        return (doc.metadata.get("page", -1), hashlib.sha1(head.encode("utf-8", "ignore")).hexdigest())
    for lst in lists:
        for item in lst:
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
            else:
                doc, score = item, None
            key = sig(doc)
            if key in seen:
                continue
            seen.add(key)
            merged.append((doc, score))
    def sort_key(t):
        doc, score = t
        page = doc.metadata.get("page", 9999)
        s = score if isinstance(score, (int, float)) else 9.99
        return (s + 0.02 * max(0, page), page)
    merged.sort(key=sort_key)
    return merged[:max_items]

def _expanded_queries(q: str):
    ql = q.lower()
    extras = []
    if any(w in ql for w in ["tariff", "payment", "price", "billing", "invoice"]):
        extras += ["tariff structure", "billing terms", "payment obligations", "invoice procedure"]
    if any(w in ql for w in ["term", "duration", "expiry", "tenure", "renewal"]):
        extras += ["term of agreement", "expiry date", "initial term", "renewal period"]
    if any(w in ql for w in ["termination", "default", "breach", "remedy"]):
        extras += ["termination events", "events of default", "remedies", "liquidated damages"]
    if any(w in ql for w in ["rights", "obligations", "duties", "responsibility", "liability"]):
        extras += ["obligations of seller", "obligations of purchaser", "rights of parties"]
    if any(w in ql for w in ["law", "jurisdiction", "arbitration", "dispute"]):
        extras += ["governing law", "dispute resolution", "arbitration clause"]
    return list(dict.fromkeys([q] + extras))

# ------------------ RAG Answer ------------------
def rag_answer(query: str, k: int = 8, max_context_chars: int = 12000, debug: bool = False):
    # Base similarity search (Chroma will embed the query under the hood)
    base = vectordb.similarity_search_with_score(query, k=k)

    # Prioritize early pages for definitions/recitals
    early = []
    if any(w in query.lower() for w in ["recital", "definition", "interpretation", "definition of"]):
        try:
            early = vectordb.similarity_search_with_score(query, k=k, filter={"page": {"$lte": 3}})
        except Exception:
            early = []

    # Expanded queries to capture clause wording
    expanded_hits = []
    for q2 in _expanded_queries(query):
        if q2 == query:
            continue
        try:
            expanded_hits += vectordb.similarity_search_with_score(q2, k=max(4, k//2))
        except Exception:
            pass

    # MMR for diverse context
    try:
        mmr_docs = vectordb.max_marginal_relevance_search(query, k=max(6, k), fetch_k=24, lambda_mult=0.3)
    except Exception:
        mmr_docs = []

    merged = _merge_results_with_scores(base, early, expanded_hits, mmr_docs, max_items=14)

    # Build context but keep it bounded in characters
    context_parts, used = [], 0
    for doc, score in merged:
        piece_body = doc.page_content.strip() if doc.page_content else ""
        piece = f"(Page {doc.metadata.get('page','NA')}) {piece_body}"
        if used + len(piece) > max_context_chars:
            break
        context_parts.append(piece)
        used += len(piece)

    context = "\n\n---\n\n".join(context_parts) if context_parts else "NO MATCHING CONTEXT FOUND."

    prompt = f"""You are a contract analysis assistant.
Answer the user’s question *only* using the context below.
If the answer is not contained in the context, reply: "The contract does not specify."
Do not invent or generalize.
Cite relevant page numbers inline in your answer.

Question: {query}
Context: {context}
Answer (clear, concise, grounded in context):""".strip()

    # Call Gemini
    try:
        resp = model.generate_content(prompt)
        # genai responses can expose text differently across SDK versions
        answer_text = ""
        # try common attributes
        if hasattr(resp, "text") and resp.text:
            answer_text = resp.text.strip()
        elif hasattr(resp, "candidates") and resp.candidates:
            # candidates may be a list of objects with 'content' or 'text'
            c0 = resp.candidates[0]
            answer_text = getattr(c0, "content", None) or getattr(c0, "text", "") or ""
            answer_text = answer_text.strip()
        else:
            # generic fallback: try str()
            answer_text = str(resp)[:2000].strip()
        if not answer_text:
            answer_text = "I couldn't find that in the document."
    except Exception as e:
        answer_text = f"[Gemini error] {e}"

    # Build sources for traceability
    sources = [
        {
            "score": float(score) if score is not None else None,
            "page": doc.metadata.get("page"),
            "preview": (doc.page_content or "")[:240].replace("\n", " "),
        }
        for doc, score in merged
    ]
    if debug:
        return answer_text, sources, {"prompt": prompt, "merged_count": len(merged)}
    return answer_text, sources

# ------------------ CLI Loop ------------------
def main():
    from collections import deque
    chat_history = deque(maxlen=5)

    print("\n📑 PPA Chatbot Ready!")
    print("Ask about clauses (e.g., Term, Tariff, Termination, Obligations). Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ["quit", "exit", "bye"]:
            print("👋 Goodbye!")
            break
        history_text = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in chat_history])
        full_query = f"{history_text}\n\nUser: {q}" if history_text else q
        ans, srcs = rag_answer(q, k=8, debug=False)
        print("\nBot:", ans, "\n")
        print("Sources:")
        for s in srcs[:5]:
            score = s["score"] if s["score"] is not None else "mmr"
            print(f"  score={score} | page={s['page']} | {s['preview'][:120]}...")
        print("\n" + "=" * 100 + "\n")
        chat_history.append((q, ans))


if __name__ == "__main__":
    main()
