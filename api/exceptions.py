import fastapi

class ClassificationError(fastapi.HTTPException):
    def __init__(self, detail: str = "Classification failed"):
        super().__init__(status_code=500, detail=detail)