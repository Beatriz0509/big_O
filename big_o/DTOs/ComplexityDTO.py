from DTOs.BaseDTO import BaseDTO
from DTOs.ClassDTO import ClassDTO
from typing import List

class ComplexityDTO(BaseDTO):
    class_: ClassDTO = ClassDTO()
    executionTime: List[float] = []
    meanExecutionTime: float = 0.0
    standardDeviation: float = 0.0
    residual: float = 0.0
    best: bool = False

    class Config:
        from_attributes = True