from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_db
from app.dto.response import JobStatusResponse, JobDetailResponse
from app.services.job_service import JobService
from app.services.queue_service import queue_service
from app.core.health import get_health_status
from app.core.exceptions import JobNotFoundException, QueueFullException
from app.core.logger import get_logger
import asyncio

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/submit",
    status_code=status.HTTP_201_CREATED,
    summary="Submit an object for processing",
    description="Upload an image file (PNG, JPG, etc.) for OCR and data extraction processing. The file will be processed through multiple stages including checksum, OCR, stamp detection, and data extraction.",
    response_description="Returns the job ID and status indicating the job has been queued"
)
async def submit_object(
    email: str = Form(
        default="sourav@gmail.com",
        description="Email address of the user submitting the job",
        example="sourav@gmail.com"
    ),
    processor: str = Form(
        default="online-something",
        description="Processor type to use for the job",
        example="online-something"
    ),
    object_file: UploadFile = File(
        ...,
        description="Image file to process (PNG, JPG, JPEG, etc.). The uploaded file will be visible in Swagger UI.",
        example="receipt.png"
    ),
    db: AsyncSession = Depends(get_db)
):
    try:
        if not object_file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file. Filename is required."
            )
        
        object_data = await object_file.read()
        if len(object_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file"
            )
        
        job = await JobService.create_job(
            session=db,
            email=email,
            object_name=object_file.filename,
            object_data=object_data,
            processor=processor
        )
        
        try:
            await queue_service.enqueue(job.id)
            logger.info(f"Job {job.id} queued successfully for email {email}")
        except QueueFullException:
            logger.error(f"Queue full, cannot enqueue job {job.id}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Processing queue is full. Please try again later."
            )
        
        return {"job_id": job.id, "status": "queued"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting object: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit object for processing"
        )


@router.get("/status/{email}", response_model=list[JobStatusResponse])
async def get_status_by_email(
    email: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        jobs = await JobService.get_jobs_by_email(db, email)
        logger.info(f"Retrieved {len(jobs)} jobs for email {email}")
        return jobs
    except Exception as e:
        logger.error(f"Error retrieving jobs for email {email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job status"
        )


@router.get("/job/{job_id}", response_model=JobDetailResponse)
async def get_job_details(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        job = await JobService.get_job_by_id(db, job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job details"
        )


@router.get("/health")
async def health_check():
    return await get_health_status()
