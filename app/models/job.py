from sqlalchemy import Column, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from .base import Base
from app.enums import JobStatus
import uuid


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, nullable=False, index=True)
    original_filename = Column(String, nullable=False)
    object_filename = Column(String, nullable=False)
    object_metadata = Column(JSON, nullable=True)
    processor = Column(String, nullable=False)
    status = Column(String, nullable=False, default=JobStatus.IN_QUEUE.value)
    total_time_taken = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
