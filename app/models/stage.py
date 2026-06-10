from sqlalchemy import Column, String, Text, Float, DateTime, JSON
from sqlalchemy.sql import func
from .base import Base
from app.enums import StageStatus
import uuid


class Stage(Base):
    __tablename__ = "stages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, nullable=False, index=True)
    stage_name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    output = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)
    params = Column(JSON, nullable=True, default=list)
    execution_time = Column(Float, nullable=True)
    llm_prompt = Column(Text, nullable=True)
    llm_response = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
