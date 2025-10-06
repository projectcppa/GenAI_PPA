import os
import re
import json
import torch
from collections import defaultdict
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.docstore.document import Document

# ------------------------
# Config
# ------------------------
GOOGLE_API_KEY = "AIzaSyAQ67pr0uC5nwlfuJorkIonbmW0QgIclWU"
CHROMA_DIR = r"C:\Chroma"
ABBREV_FILE = os.path.join(CHROMA_DIR, "abbreviations.json")

# ------------------------
# Load Embeddings + Vectorstore
# ------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# ------------------------
# Load Abbreviation Index
# ------------------------
if not os.path.exists(ABBREV_FILE):
    raise FileNotFoundError(f"❌ Abbreviation file not found at {ABBREV_FILE}. Run ingestion first.")

with open(ABBREV_FILE, "r", encoding="utf-8") as f:
    ABBREV_INDEX = json.load(f)

# ------------------------
# LLM + Memory
# ------------------------
chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ------------------------
# Hybrid Retrieval
# ------------------------
def hybrid_retrieval(query, vectorstore, k=20):
    results = vectorstore.similarity_search(query, k=k)

    # Find abbreviations in query
    matches = re.findall(r"\b[A-Z]{2,}[a-zA-Z0-9]*[a-z]?\b", query)
    for m in matches:
        if m in ABBREV_INDEX:
            print(f"🔍 Direct abbreviation lookup: {m}")
            for entry in ABBREV_INDEX[m]:
                # Convert back into Document for uniformity
                results.append(Document(
                    page_content=entry["text"],
                    metadata={"source": entry["source"], "page": entry["page"]}
                ))

    # Deduplicate
    unique = {}
    for doc in results:
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:50])
        unique[key] = doc
    return list(unique.values())

# ------------------------
# Query Rewriting
# ------------------------
query_rephrase_prompt = PromptTemplate.from_template("""
Given conversation history and a follow-up question,
rephrase it into a standalone question. If it's already standalone, return as-is.

Chat History:
{chat_history}
Follow Up Question: {question}
Standalone question:"""
)

query_rephrase_chain = (
    RunnablePassthrough.assign(
        chat_history=lambda x: memory.load_memory_variables({})['chat_history']
    )
    | query_rephrase_prompt
    | chat
    | StrOutputParser()
)

# ------------------------
# Main Loop
# ------------------------
print("\n📚 CPPA Document Assistant\n")

while True:
    query = input("Ask a question: ")
    if query.lower() in {"exit", "quit"}:
        break

    rephrased_query = query_rephrase_chain.invoke({"question": query})
    print(f"🔄 Rephrased query: {rephrased_query}")

    docs = hybrid_retrieval(rephrased_query, vectorstore, k=25)

    if not docs:
        print("Apologies! I could not find this information in the documents.")
        continue

    grouped = defaultdict(list)
    for doc in docs:
        filename = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")
        grouped[(filename, page)].append(doc.page_content)

    context_parts = []
    for (filename, page), chunks in grouped.items():
        src = f"{filename} (Page {page})"
        full_text = "\n".join(chunks)
        context_parts.append(f"[{src}]\n{full_text}")

    context = "\n\n".join(context_parts)

    messages = [
        ("system",
         "You are an expert assistant specialized in analyzing multiple documents.\n"
         "Guidelines:\n"
         "1. Use ONLY the provided content. Never invent or generalize.\n"
         "2. If the query mentions a specific document, restrict results to that document only.\n"
         "3. If the same information exists in multiple docs, show them separately with source (doc + page).\n"
         "4. If asked to compare, highlight similarities and differences clearly (bullets or table).\n"
         "5. If no content matches, respond exactly:\n"
         "   'Apologies! I could not find this information in the documents.'\n"
         "6. Expand abbreviations if present in the query and explain them if found in docs.\n"
         "7. Always return formulas exactly as they appear in the document.\n"
         "8. Keep the structure and variables identical.\n"
         "9. Keep answers concise, factual, and professional.\n"
         "10. If multiple related abbreviations (e.g. NEO, NEOm, NEOh) exist, you MUST return ALL of them.\n"
        "11. Always list each abbreviation as a separate bullet with its exact definition and source (doc + page).\n"
        "12. Never omit any abbreviation that appears in the provided context, even if the user only asked for the base form.\n"),

        ("human", f"Context:\n{context}\n\nQuestion:\n{query}")
    ]

    print("\n🤖 Thinking...\n")
    response = chat.invoke(messages).content
    print("📄 Answer:\n", response)
