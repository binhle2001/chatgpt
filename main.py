from fastapi.responses import JSONResponse
from search import *
from add_accent import add_accent
from fastapi import FastAPI, status
from schema import Message
from ai_configs import ERROR_MESSAGE

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
    response, _ = get_response(
            query=message,
            df=embedding_data,
            model="gpt-3.5-turbo",
        )   
    if ERROR_MESSAGE in response:
        request = add_accent(message)
        response_, _ = get_response(
            query=request,
            df=embedding_data,
            model="gpt-3.5-turbo",
        )   
        if ERROR_MESSAGE not in response_:
            content = {
                "http_code": status.HTTP_200_OK,
                "data": response_
            }
            return JSONResponse(status_code = status.HTTP_200_OK, content = content
)
    content = {
                "http_code": status.HTTP_200_OK,
                "data": response_
            }
           
    return JSONResponse(status_code = status.HTTP_200_OK, content = content
)
