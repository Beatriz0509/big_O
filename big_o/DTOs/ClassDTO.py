# Modified by Tiago Fonseca (@tiagosf13), 2025

from DTOs.BaseDTO import BaseDTO

class ClassDTO(BaseDTO):
    name: str = ""

    class Config:
        from_attributes = True