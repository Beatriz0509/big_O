from DTOs.BaseDTO import BaseDTO
from DTOs.InputDTO import InputDTO
from DTOs.OutputDTO import OutputDTO

class TestDTO(BaseDTO):
    input: InputDTO = InputDTO()
    output: OutputDTO = OutputDTO()
    executionTime: float = 0.0

    class Config:
        from_attributes = True