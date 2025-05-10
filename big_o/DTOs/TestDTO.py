from src.services.BigO.DTOs.BaseDTO import BaseDTO
from src.services.BigO.DTOs.InputDTO import InputDTO
from src.services.BigO.DTOs.OutputDTO import OutputDTO

class TestDTO(BaseDTO):
    input: InputDTO = InputDTO()
    output: OutputDTO = OutputDTO()
    executionTime: float = 0.0

    class Config:
        from_attributes = True