from pydantic import BaseModel
from typing import Optional
from DTOs.ResponseHealthDTO import ResponseHealthDTO

class BaseDTO(BaseModel):
    uuid: Optional[str] = ""
    deleted: bool = False
    status: Optional[ResponseHealthDTO] = ResponseHealthDTO()

    class Config:
        from_attributes = True