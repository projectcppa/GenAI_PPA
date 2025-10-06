import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import torch

# ------------------------
# Configuration
# ------------------------
GOOGLE_API_KEY = "AIzaSyB4ixfEQ-4Iv2hfUjM_3GOm6-plaDtnzSE"  # Replace with your key or load from env
FAISS_DIR = "faiss_db"

# ------------------------
# Load Embeddings & Vector Store
# ------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

vectorstore = FAISS.load_local(
    FAISS_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

# ------------------------
# Set up Google Gemini
# ------------------------
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# ------------------------
# Chat History
# ------------------------
chat_history = []

# ------------------------
# Query Loop
# ------------------------
print("📚 CPPA Document Assistant (with memory + relevance check)")
print("Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ")
    if query.lower() in {"exit", "quit"}:
        break

    # ------------------------
    # Step 1: Decide if new question is related to history
    # ------------------------
    history_text = "\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in chat_history]
    )

    if chat_history:  # only check if there is some history
        relevance_prompt = f"""
Determine if the new user question is related to the previous conversation. 
Answer with only "Yes" or "No".

Conversation so far:
{history_text}

New question:
{query}
"""
        decision = chat.invoke(relevance_prompt).content.strip().lower()
    else:
        decision = "no"

    use_history = "yes" in decision

    # ------------------------
    # Step 2: Build retrieval query
    # ------------------------
    if use_history:
        # Include last 10 turns for retrieval
        last_exchanges = " ".join(
            [f"User: {q} Assistant: {a}" for q, a in chat_history[-10:]]
        )
        retrieval_query = f"{last_exchanges} {query}"
    else:
        retrieval_query = query

    docs = vectorstore.similarity_search(retrieval_query, k=4)

    # ------------------------
    # Step 3: Prepare context
    # ------------------------
    context_parts, sources = [], set()
    for doc in docs:
        filename = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")
        src = f"{filename} (Page {page})"
        if src not in sources:
            context_parts.append(f"[{src}]\n{doc.page_content}")
            sources.add(src)

    context = "\n\n".join(context_parts)

    # ------------------------
    # Step 4: Build final prompt
    # ------------------------
    prompt = f"""
You are an expert in analyzing multiple documents.
Use ONLY the content provided below to answer.
If the same answer appears in multiple documents, show it separately for each doc with its source.
If the query asks for a comparison, highlight differences across documents.
If no document contains the answer, reply:
"Apologies! I could not find this information in the documents."
Do not invent or generalize. Be accurate.

Conversation so far:
{history_text if use_history else "(history skipped for this question)"}

Context:
{context}

Question:
{query}
"""

    # ------------------------
    # Step 5: Generate response
    # ------------------------
    print("\n🤖 Thinking...\n")
    response = chat.invoke(prompt).content
    print("📄 Answer:\n")
    print(response)

    

    # ------------------------
    # Step 6: Save to history
    # ------------------------
    chat_history.append((query, response))
    chat_history = chat_history[-20:]  # keep last 20 turns
