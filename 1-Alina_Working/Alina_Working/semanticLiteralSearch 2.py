import os
import re
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from collections import defaultdict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------------
# Configuration
# ------------------------
GOOGLE_API_KEY = "AIzaSyB4ixfEQ-4Iv2hfUjM_3GOm6-plaDtnzSE"
FAISS_DIR = "faiss_db"

# ------------------------
# Load Embeddings & FAISS
# ------------------------
try:
    print("⏳ Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    print(f"⏳ Loading FAISS index from {FAISS_DIR}...")
    
    if not os.path.exists(FAISS_DIR) or not any(f.endswith(".faiss") or f.endswith(".pkl") for f in os.listdir(FAISS_DIR)):
        raise FileNotFoundError(f"FAISS index not found in directory: {FAISS_DIR}. Please ensure you have created and saved the index first.")

    vectorstore = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    docs = list(vectorstore.docstore._dict.values())
    print("✅ FAISS index and documents loaded.")

except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    print("Please ensure your FAISS index is properly built and located in the correct directory.")
    exit()

# ------------------------
# Set up Google Gemini & Conversation Memory
# ------------------------
print("⏳ Initializing LLM and conversation memory...")
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
print("✅ Conversation memory initialized.")

# ------------------------
# Query Rewriting Chain
# ------------------------
# This chain rephrases the user's query into a standalone question
query_rephrase_prompt = PromptTemplate.from_template("""
Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question.
If the question is already a standalone question, return it as is.
Do NOT try to answer the question, just rephrase it.

Chat History:
{chat_history}
Follow Up Question: {question}
Standalone question:"""
)

query_rephrase_chain = (
    RunnablePassthrough.assign(chat_history=lambda x: memory.load_memory_variables({})['chat_history'])
    | query_rephrase_prompt
    | chat
    | StrOutputParser()
)

# ------------------------
# Query Helper - Literal Filter Extractor
# ------------------------
def extract_filters(query):
    """
    Extracts specific filters (page, section, file) from a query string.
    """
    filters = {}
    
    page_match = re.search(r'page\s+(\d+)', query, re.IGNORECASE)
    if page_match:
        filters['page'] = int(page_match.group(1))

    section_match = re.search(r'section\s+([\d\.]+)', query, re.IGNORECASE)
    if section_match:
        filters['section'] = section_match.group(1).lower()

    schedule_match = re.search(r'schedule\s+(\d+)', query, re.IGNORECASE)
    if schedule_match:
        filters['schedule'] = schedule_match.group(1)

    file_match = re.search(r'([a-zA-Z0-9_\- ]+\.pdf)', query, re.IGNORECASE)
    if file_match:
        filters['source'] = file_match.group(1).strip().lower()

    return filters

# ------------------------
# Main Query Loop
# ------------------------
print("\n📚 CPPA Document Assistant")
print("I'm ready to answer your questions. Type 'exit' to quit.\n")

while True:
    try:
        query = input("Ask a question: ")
        if query.lower() in {"exit", "quit"}:
            break
        
        # Use the query rewriting chain to get a standalone query for search
        rephrased_query = query_rephrase_chain.invoke({"question": query})
        print(f"🔄 Rephrased query for search: {rephrased_query}")

        filters = extract_filters(rephrased_query)
        filtered_docs = docs

        # Apply literal filters
        if filters:
            print("🔍 Applying literal filters...")
            if 'source' in filters:
                filtered_docs = [doc for doc in filtered_docs if filters['source'] in doc.metadata.get("source", "").lower()]
            
            if 'page' in filters:
                filtered_docs = [doc for doc in filtered_docs if doc.metadata.get("page") == filters['page']]
            
            if 'section' in filters:
                filtered_docs = [doc for doc in filtered_docs if filters['section'] in doc.page_content.lower()]

            if 'schedule' in filters:
                filtered_docs = [doc for doc in filtered_docs if f"schedule {filters['schedule']}" in doc.page_content.lower()]

        # Fallback to semantic search if no literal matches or filters
        if not filters or len(filtered_docs) == 0:
            print("🔍 Using semantic search...")
            filtered_docs = vectorstore.similarity_search(rephrased_query, k=4)
            
        # Group chunks by source + page
        grouped = defaultdict(list)
        for doc in filtered_docs:
            filename = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "?")
            key = (filename, page)
            grouped[key].append(doc.page_content)

        # Merge chunks into full pages
        context_parts, sources = [], set()
        for (filename, page), chunks in grouped.items():
            src = f"{filename} (Page {page})"
            full_page_text = "\n".join(chunks)
            context_parts.append(f"[{src}]\n{full_page_text}")
            sources.add(src)

        context = "\n\n".join(context_parts)
        
        if not context.strip():
            print("⚠️ No matching content found.")
            response = "No matching content found."
            memory.save_context({"input": query}, {"output": response})
            print(f"📄 Answer:\n{response}")
            continue

        # Build the final prompt with conversation history and context
        chat_history_messages = memory.load_memory_variables({})['chat_history']
        messages = [
            ("system", "You are an expert in analyzing multiple documents. "
            "Use ONLY the content provided below to answer. "
            "If the same answer appears in multiple documents, show it separately for each doc with its source. "
            "If the query asks for a comparison, highlight differences across documents. "
            "If no document contains the answer, reply: "
            "'Apologies! I could not find this information in the documents.' "
            "Do not invent or generalize. Be accurate."),
        ]
        
        for msg in chat_history_messages:
            messages.append((msg.type, msg.content))
        
        messages.append(("human", f"Context:\n{context}\n\nQuestion:\n{query}"))
        
        # Generate answer
        print("\n🤖 Thinking...\n")
        response = chat.invoke(messages).content
        print("📄 Answer:")
        print(response)
        
        # Save the current exchange to memory for the next turn
        memory.save_context({"input": query}, {"output": response})
        
    except Exception as e:
        print(f"❌ An error occurred while processing your query: {e}")
    finally:
        print("-" * 80)
