# single_agent.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Single File AI Agent")

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    message = req.message.lower()
    if "hello" in message:
        reply = "Hello! I am your AI agent."
    elif "how are you" in message:
        reply = "I'm just code, running perfectly!"
    else:
        reply = f"[Stub reply] You said: {req.message}"
    return ChatResponse(session_id=req.session_id, reply=reply)
