from pydantic import BaseModel, EmailStr


class ObjectSubmissionRequest(BaseModel):
    email: EmailStr
    processor: str
