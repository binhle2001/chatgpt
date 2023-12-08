from search import get_response
from add_accent import add_accent
from fastapi import FastAPI
from schema import Message
app = FastAPI(
    "khanhdo",
    summary="chatbot base on chatgpt of Khanh Do",
    version="1.0.0"
)

@app.post(
    "/chat"
)
async def get_response_from_chatgpt(
    message: Message
):
    response 