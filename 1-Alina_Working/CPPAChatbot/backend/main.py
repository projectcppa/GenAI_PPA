from fastapi import FastAPI
from ingest import run_ingestion
from chat_service import ask_question, get_history
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CPPA Document Assistant")

# Allow all origins (for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.post("/ingest")
def ingest_docs():
    result = run_ingestion()
    return {"message": "Ingestion completed", "details": result}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    answer = ask_question(request.query)
    return {"question": request.query, "answer": answer}

@app.get("/history")
def history_endpoint():
    return {"history": get_history()}

 