from DTOs.BaseDTO import BaseDTO

class InputDTO(BaseDTO):
    value: str = ""

    class Config:
        from_attributes = True