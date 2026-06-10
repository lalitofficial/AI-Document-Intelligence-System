from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class StageResponse(BaseModel):
    stage_name: str
    status: str
    output: Optional[str] = None
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    params: List[str]
    execution_time: Optional[float] = None


class JobStatusResponse(BaseModel):
    email: str
    object_name: str
    status: str
    stages: List[StageResponse]


class JobDetailResponse(BaseModel):
    job_id: str
    email: str
    object_name: str
    status: str
    stages: List[StageResponse]
    total_time_taken: Optional[float] = None
    created_at: datetime
    updated_at: datetime
