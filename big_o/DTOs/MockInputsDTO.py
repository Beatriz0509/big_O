from pydantic import BaseModel


class MockInputsDTO(BaseModel):
    min_n: int
    max_n: int

    class Config:
        from_attributes = True