# ------------------ Targeted RAG over Power Purchase Agreement (PPA) ------------------
# - Unique Chroma store per (file hash, embedding model)
# - Falls back to OCR if no extractable text
# - Cleans headers/footers typical in contracts
# - Uses PPA-specific query expansion (clauses, obligations, tariff, termination)
# - Builds grounded prompt for Gemini
# -------------------------------------------------------------------------------------

import os, hashlib, re, io
from pathlib import Path
from dotenv import load_dotenv

# 1) Keys
load_dotenv("token.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Put it in token.env as GEMINI_API_KEY=...")

# 2) Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import pytesseract

# Point to your installed Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\mahnoor.shakir\tesseract.exe"

# ------------------ Config ------------------
PDF_PATH = r"C:\Users\mahnoor.shakir\Downloads\karot.pdf"
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
    """Remove headers/footers & boilerplate common in PPAs."""
    lines = txt.splitlines()
    drop_patterns = [
        r"^Page\s*\d+", r"^Power Purchase Agreement", r"^Confidential",
        r"^Draft", r"^Executed Version", r"^This Power Purchase Agreement",
        r"^Table of Contents", r"^\s*PPA\s*$"
    ]
    drop_re = re.compile("|".join(drop_patterns), re.IGNORECASE)
    kept = [ln for ln in lines if ln.strip() and not drop_re.search(ln)]
    text = re.sub(r"\s+\n", "\n", "\n".join(kept))
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def load_pdf_with_ocr(path: str):
    """Try normal text extraction; fallback to OCR if needed."""
    loader = PyPDFLoader(path)
    docs = loader.load()

    if all(not d.page_content.strip() for d in docs):
        print("⚠️ No extractable text found, running OCR...")
        pdf = fitz.open(path)
        ocr_docs = []
        for i, page in enumerate(pdf):
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="eng")
            ocr_docs.append(
                Document(page_content=text, metadata={"source": path, "page": i+1})
            )
        return ocr_docs
    return docs

# ------------------ Unique Persist Path ------------------
file_hash = sha1_of_file(PDF_PATH)
file_tag  = sanitize(Path(PDF_PATH).stem.lower())
embed_tag = sanitize(EMBED_MODEL.replace("/", "_"))
PERSIST_DIR = f"./chroma_store/{file_tag}/{embed_tag}/{file_hash[:12]}"
COLLECTION_NAME = f"{file_tag}_{embed_tag}"

# ------------------ Load, Clean & Chunk ------------------
docs = load_pdf_with_ocr(PDF_PATH)

# Clean page text + ensure integer page metadata
for d in docs:
    d.page_content = clean_text(d.page_content)
    if "page" in d.metadata:
        try:
            d.metadata["page"] = int(d.metadata["page"])
        except Exception:
            pass

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=120,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"Loaded {len(chunks)} cleaned chunks from {PDF_PATH}")

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
        head = doc.page_content[:120].strip()
        return (doc.metadata.get("page", -1), hashlib.sha1(head.encode("utf-8", "ignore")).hexdigest())
    for lst in lists:
        for item in lst:
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
            else:
                doc, score = item, None
            key = sig(doc)
            if key in seen: continue
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
    if any(w in ql for w in ["tariff", "payment", "price", "billing"]):
        extras += ["tariff structure", "billing terms", "payment obligations", "invoice procedure"]
    if any(w in ql for w in ["term", "duration", "expiry", "tenure"]):
        extras += ["term of agreement", "expiry date", "initial term", "renewal period"]
    if any(w in ql for w in ["termination", "default", "breach"]):
        extras += ["termination events", "events of default", "remedies", "liquidated damages"]
    if any(w in ql for w in ["rights", "obligations", "duties", "responsibility"]):
        extras += ["obligations of seller", "obligations of purchaser", "rights of parties"]
    if any(w in ql for w in ["law", "jurisdiction", "arbitration", "dispute"]):
        extras += ["governing law", "dispute resolution", "arbitration clause"]
    return list(dict.fromkeys([q] + extras))

# ------------------ RAG Answer ------------------
def rag_answer(query: str, k: int = 8, max_context_chars: int = 12000, debug: bool = False):
    base = vectordb.similarity_search_with_score(query, k=k)
    early = []
    if any(w in query.lower() for w in ["recital", "definitions", "interpretation"]):
        try:
            early = vectordb.similarity_search_with_score(query, k=k, filter={"page": {"$lte": 3}})
        except Exception:
            pass
    expanded_hits = []
    for q2 in _expanded_queries(query):
        if q2 == query: continue
        expanded_hits += vectordb.similarity_search_with_score(q2, k=max(4, k//2))
    try:
        mmr_docs = vectordb.max_marginal_relevance_search(query, k=max(6, k), fetch_k=24, lambda_mult=0.3)
    except Exception:
        mmr_docs = []
    merged = _merge_results_with_scores(base, early, expanded_hits, mmr_docs, max_items=14)

    context_parts, used = [], 0
    for doc, score in merged:
        piece = f"[score={0.0 if score is None else score:.4f} | page={doc.metadata.get('page','NA')}]\n{doc.page_content.strip()}"
        if used + len(piece) > max_context_chars: break
        context_parts.append(piece); used += len(piece)
    context = "\n\n---\n\n".join(context_parts) if context_parts else "NO MATCHING CONTEXT FOUND."

    prompt = f"""
You are a contract analysis assistant.
Answer ONLY using the CONTEXT from the Power Purchase Agreement (PPA).
If the answer is not in the context, say: "I couldn't find that in the document."
Always point to the clause or phrase exactly as written.

CONTEXT:
{context}

QUESTION: {query}

RESPONSE (precise, clause-based):
""".strip()

    try:
        resp = model.generate_content(prompt)
        answer_text = (getattr(resp, "text", "") or "").strip()
        if not answer_text: answer_text = "I couldn't find that in the document."
    except Exception as e:
        answer_text = f"[Gemini error] {e}"

    sources = [
        {"score": float(score) if score is not None else None,
         "page": doc.metadata.get("page"),
         "preview": doc.page_content[:240].replace("\n", " ")}
        for doc, score in merged
    ]
    return answer_text, sources

# ------------------ CLI Loop ------------------
from collections import deque
chat_history = deque(maxlen=5)

print("\n📑 PPA Chatbot Ready!")
print("Ask me about clauses (e.g., Term, Tariff, Termination, Obligations). Type 'quit' to exit.\n")

while True:
    q = input("You: ").strip()
    if q.lower() in ["quit", "exit", "bye"]:
        print("👋 Goodbye!")
        break
    history_text = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in chat_history])
    full_query = f"{history_text}\n\nUser: {q}" if history_text else q
    ans, srcs = rag_answer(full_query, k=8, debug=False)

    print("\nBot:", ans)
    print("Sources:")
    for s in srcs[:5]:
        score = s['score'] if s['score'] is not None else 'mmr'
        print(f"  score={score} | page={s['page']} | {s['preview'][:120]}...")

    chat_history.append((q, ans))
    print("\n" + "="*100 + "\n")