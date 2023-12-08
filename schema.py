from pydantic import BaseModel,  Field

class Message(BaseModel):
    message: str = Field(examples="")
    class Config:
        schema_extra = {
            "message": ""
        }