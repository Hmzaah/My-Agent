from fastapi import FastAPI
from pydantic import BaseModel
from app.llm_adapter import LLMAdapter

app = FastAPI(title="My Modular AI Agent")

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str

llm_adapter = LLMAdapter()

def simple_agent(message: str) -> str:
    return llm_adapter.generate(message)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    reply_text = simple_agent(req.message)
    return ChatResponse(session_id=req.session_id, reply=reply_text)
