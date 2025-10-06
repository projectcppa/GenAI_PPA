import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
 
GOOGLE_API_KEY = "AIzaSyAQ67pr0uC5nwlfuJorkIonbmW0QgIclWU"
HISTORY_FILE = "chat_history.json"
 
# Load old history
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
else:
    history = []
 
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)
 
print("💬 Chat loaded. Type 'exit' to quit.\n")
 
# 👉 Show past history in terminal
if history:
    print("📜 Previous conversation:")
    for turn in history:
        print(f"You: {turn['input']['input']}")
        print(f"🤖 Bot: {turn['output']['output']}\n")
    print("-" * 60)
else:
    print("🆕 No previous chat history found.")
    print("-" * 60)
 
# Chat loop
while True:
    query = input("You: ")
    if query.lower() in {"exit", "quit"}:
        break
 
    # Build messages for Gemini
    messages = [("system", "You are a helpful assistant.")]
    for turn in history:
        messages.append(("human", turn["input"]["input"]))
        messages.append(("ai", turn["output"]["output"]))
    messages.append(("human", query))
 
    # Get response
    response = chat.invoke(messages).content
    print(f"🤖 Bot: {response}\n")
 
    # Save to history
    history.append({"input": {"input": query}, "output": {"output": response}})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)