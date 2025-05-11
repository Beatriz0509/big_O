# Modified by Tiago Fonseca (@tiagosf13), 2025

from pydantic import BaseModel


class MockInputsDTO(BaseModel):
    min_n: int
    max_n: int

    class Config:
        from_attributes = True