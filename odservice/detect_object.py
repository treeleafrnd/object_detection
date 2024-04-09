from typing import List
from pydantic import BaseModel

class DetectedObjectResult(BaseModel):
    bbox: List[int] = []
    confidence: float = 0
    class_name: str = None
