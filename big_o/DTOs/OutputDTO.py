from DTOs.BaseDTO import BaseDTO

class OutputDTO(BaseDTO):
    value: str = ""

    class Config:
        from_attributes = True