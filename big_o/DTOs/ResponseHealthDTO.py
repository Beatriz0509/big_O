# Modified by Tiago Fonseca (@tiagosf13), 2025

from pydantic import BaseModel
from typing import Optional

class ResponseHealthDTO(BaseModel):
    successfull: bool = True
    message: str = ""
    code: int = 0
    custom_code: Optional[int] = 0