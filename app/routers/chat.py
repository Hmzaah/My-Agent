from fastapi import APIRouter
from app.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()

@router.post("/chat")
async def chat(prompt: str):
    response = chat_service.generate_response(prompt)
    return {"response": response}
