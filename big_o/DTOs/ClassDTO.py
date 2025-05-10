from DTOs.BaseDTO import BaseDTO

class ClassDTO(BaseDTO):
    name: str = ""

    class Config:
        from_attributes = True