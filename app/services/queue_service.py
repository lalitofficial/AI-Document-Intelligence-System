import asyncio
from typing import Dict, List, Callable
from app.core.config import settings
from app.core.context import ProcessingContext
from app.core.logger import get_logger
from app.models import Job, Stage
from app.enums import JobStatus, StageStatus, ProcessorStatus, ProcessorType
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
from pathlib import Path
import aiofiles
import time

logger = get_logger(__name__)


class QueueService:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=settings.max_concurrent_jobs)
        self.workers: List[asyncio.Task] = []
        self.processor_registry: Dict[str, List[Callable]] = {}
        self._running = False

    def register_processor(self, processor_type: str, processors: List[Callable]):
        self.processor_registry[processor_type] = processors

    async def start_workers(self, db_session_factory):
        if self._running:
            return
        
        self._running = True
        for i in range(settings.max_workers):
            worker = asyncio.create_task(self._worker(db_session_factory, i))
            self.workers.append(worker)

    async def stop_workers(self):
        self._running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

    async def enqueue(self, job_id: str):
        try:
            await asyncio.wait_for(self.queue.put(job_id), timeout=5.0)
            logger.info(f"Job {job_id} enqueued successfully")
        except asyncio.TimeoutError:
            logger.error(f"Timeout while enqueuing job {job_id}")
            from app.core.exceptions import QueueFullException
            raise QueueFullException(f"Queue is full, cannot enqueue job {job_id}")

    async def _worker(self, db_session_factory, worker_id: int):
        while self._running:
            try:
                job_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                await self._process_job(job_id, db_session_factory)
                self.queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)

    async def _process_job(self, job_id: str, db_session_factory):
        async with db_session_factory() as session:
            try:
                from sqlalchemy import select
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                
                if not job:
                    logger.warning(f"Job {job_id} not found in database")
                    return

                logger.info(
                    f"Processing job {job_id} - "
                    f"Email: {job.email}, "
                    f"Original Object: {job.original_filename}, "
                    f"Object UUID: {job.object_filename}, "
                    f"Processor: {job.processor}, "
                    f"Metadata: {job.object_metadata}"
                )

                job.status = JobStatus.PROCESSING.value
                job.updated_at = datetime.now(timezone.utc)
                await session.commit()

                object_file_path = Path(settings.bucket_storage_folder) / job.object_filename
                if not object_file_path.exists():
                    raise FileNotFoundError(f"Object file not found: {object_file_path}")
                
                async with aiofiles.open(object_file_path, 'rb') as f:
                    object_bytes = await f.read()
                
                logger.info(f"Loaded object from filesystem: {object_file_path} (size: {len(object_bytes)} bytes)")
                
                await ProcessingContext.set(job_id, "object_data", object_bytes)
                await ProcessingContext.set(job_id, "processor_type", job.processor)
                await ProcessingContext.set(job_id, "original_filename", job.original_filename)
                await ProcessingContext.set(job_id, "object_filename", job.object_filename)
                await ProcessingContext.set(job_id, "object_metadata", job.object_metadata)
                await ProcessingContext.set(job_id, "email", job.email)

                processors = self.processor_registry.get(ProcessorType.DEFAULT.value, [])
                
                logger.info(f"Starting processing pipeline with {len(processors)} stages for job {job_id}")
                start_time = time.time()
                
                for processor in processors:
                    stage_name = processor.stage_name
                    logger.info(f"Starting stage {stage_name} for job {job_id}")
                    
                    stage = Stage(
                        job_id=job_id,
                        stage_name=stage_name,
                        status=StageStatus.PROCESSING.value,
                        params=[]
                    )
                    session.add(stage)
                    await session.commit()
                    
                    stage_start = time.time()
                    try:
                        result = await processor.execute(job_id, [])
                    except Exception as stage_exception:
                        stage_end = time.time()
                        logger.error(f"Exception during stage {stage_name} execution for job {job_id}: {stage_exception}", exc_info=True)
                        import traceback
                        stage.status = StageStatus.ERROR.value
                        stage.error = str(stage_exception)
                        stage.stack_trace = traceback.format_exc()
                        stage.execution_time = stage_end - stage_start
                        stage.updated_at = datetime.now(timezone.utc)
                        await session.commit()
                        
                        job.status = JobStatus.FAILED.value
                        job.updated_at = datetime.now(timezone.utc)
                        await session.commit()
                        await ProcessingContext.clear(job_id)
                        return
                    
                    stage_end = time.time()
                    
                    if result["status"] == ProcessorStatus.ERROR.value:
                        stage.status = StageStatus.ERROR.value
                        stage.output = None
                        stage.error = result.get("error")
                        stage.stack_trace = result.get("stack_trace")
                        stage.execution_time = stage_end - stage_start
                        stage.updated_at = datetime.now(timezone.utc)
                        await session.commit()
                        
                        logger.error(
                            f"Stage {stage_name} failed for job {job_id}: {result.get('error')}"
                        )
                        job.status = JobStatus.FAILED.value
                        job.updated_at = datetime.now(timezone.utc)
                        await session.commit()
                        await ProcessingContext.clear(job_id)
                        return
                    else:
                        stage.status = StageStatus.SUCCESS.value
                        output = result.get("output", {})
                        if isinstance(output, dict):
                            stage.output = str(output.get("data", output))
                            if stage_name == "stage5_data_extraction":
                                stage.llm_prompt = output.get("llm_prompt")
                                stage.llm_response = output.get("llm_response")
                        else:
                            stage.output = str(output)
                        stage.error = None
                        stage.stack_trace = None
                        stage.execution_time = stage_end - stage_start
                        stage.updated_at = datetime.now(timezone.utc)
                        await session.commit()
                        
                        logger.info(
                            f"Stage {stage_name} completed successfully for job {job_id} "
                            f"(execution time: {stage_end - stage_start:.2f}s)"
                        )

                end_time = time.time()
                job.status = JobStatus.SUCCESS.value
                job.total_time_taken = end_time - start_time
                job.updated_at = datetime.now(timezone.utc)
                await session.commit()
                
                logger.info(
                    f"Job {job_id} completed successfully - "
                    f"Total time: {job.total_time_taken:.2f}s, "
                    f"Original Object: {job.original_filename}"
                )
                await ProcessingContext.clear(job_id)
                
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
                try:
                    result = await session.execute(select(Job).where(Job.id == job_id))
                    job = result.scalar_one_or_none()
                    if job:
                        job.status = JobStatus.FAILED.value
                        job.updated_at = datetime.now(timezone.utc)
                        await session.commit()
                        logger.info(f"Job {job_id} marked as failed")
                except Exception as db_error:
                    logger.error(f"Failed to update job {job_id} status: {db_error}", exc_info=True)
                finally:
                    await ProcessingContext.clear(job_id)


queue_service = QueueService()
