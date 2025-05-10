from DTOs.BaseDTO import BaseDTO

class TypeDTO(BaseDTO):
    name: str = ""

    class Config:
        from_attributes = True