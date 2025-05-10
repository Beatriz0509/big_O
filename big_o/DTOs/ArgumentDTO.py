from DTOs.BaseDTO import BaseDTO
from DTOs.TypeDTO import TypeDTO

class ArgumentDTO(BaseDTO):
    name: str = ""
    firstType: TypeDTO = TypeDTO()
    type: str = ""

    class Config:
        from_attributes = True