from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
from app.models import Job, Stage
from app.dto.response import JobStatusResponse, StageResponse, JobDetailResponse
from app.enums import JobStatus
from app.core.config import settings
from app.core.logger import get_logger
import uuid
import aiofiles
from pathlib import Path

logger = get_logger(__name__)


class JobService:
    @staticmethod
    async def create_job(
        session: AsyncSession,
        email: str,
        object_name: str,
        object_data: bytes,
        processor: str
    ) -> Job:
        object_uuid = str(uuid.uuid4())
        file_extension = Path(object_name).suffix or ""
        object_filename = f"{object_uuid}{file_extension}"
        storage_path = Path(settings.bucket_storage_folder)
        object_file_path = storage_path / object_filename
        
        object_metadata = {
            "original_filename": object_name,
            "file_size": len(object_data),
            "uuid": object_uuid,
            "file_extension": file_extension
        }
        
        async with aiofiles.open(object_file_path, 'wb') as f:
            await f.write(object_data)
        
        logger.info(f"Object saved to filesystem: {object_file_path} (size: {len(object_data)} bytes)")
        
        job = Job(
            email=email,
            original_filename=object_name,
            object_filename=object_filename,
            object_metadata=object_metadata,
            status=JobStatus.IN_QUEUE.value,
            processor=processor
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job

    @staticmethod
    async def get_jobs_by_email(session: AsyncSession, email: str) -> List[JobStatusResponse]:
        result = await session.execute(select(Job).where(Job.email == email))
        jobs = result.scalars().all()
        
        responses = []
        for job in jobs:
            stages_result = await session.execute(
                select(Stage).where(Stage.job_id == job.id).order_by(Stage.created_at)
            )
            stages = stages_result.scalars().all()
            
            stage_responses = [
                StageResponse(
                    stage_name=stage.stage_name,
                    status=stage.status,
                    output=stage.output,
                    error=stage.error,
                    stack_trace=stage.stack_trace,
                    params=stage.params or [],
                    execution_time=stage.execution_time
                )
                for stage in stages
            ]
            
            responses.append(JobStatusResponse(
                email=job.email,
                object_name=job.original_filename,
                status=job.status,
                stages=stage_responses
            ))
        
        return responses

    @staticmethod
    async def get_job_by_id(session: AsyncSession, job_id: str) -> Optional[JobDetailResponse]:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        
        if not job:
            return None
        
        stages_result = await session.execute(
            select(Stage).where(Stage.job_id == job.id).order_by(Stage.created_at)
        )
        stages = stages_result.scalars().all()
        
        stage_responses = [
            StageResponse(
                stage_name=stage.stage_name,
                status=stage.status,
                output=stage.output,
                error=stage.error,
                stack_trace=stage.stack_trace,
                params=stage.params or [],
                execution_time=stage.execution_time
            )
            for stage in stages
        ]
        
        return JobDetailResponse(
            job_id=job.id,
            email=job.email,
            object_name=job.original_filename,
            status=job.status,
            stages=stage_responses,
            total_time_taken=job.total_time_taken,
            created_at=job.created_at,
            updated_at=job.updated_at
        )
