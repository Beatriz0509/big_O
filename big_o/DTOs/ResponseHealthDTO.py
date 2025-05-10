from pydantic import BaseModel
from typing import Optional

class ResponseHealthDTO(BaseModel):
    successfull: bool = True
    message: str = ""
    code: int = 0
    custom_code: Optional[int] = 0