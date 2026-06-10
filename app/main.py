from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import router
from app.db import init_db, get_database_factory
from app.services.queue_service import queue_service
from app.processors import Stage1ChecksumProcessor, Stage2SHA1Processor, Stage3OCRProcessor, Stage4StampProcessor, Stage5DataExtractionProcessor
from app.enums import ProcessorType
from app.core.logger import get_logger
from app.services.ocr_service import get_ocr_service
from app.services.stamp_service import get_stamp_service
from app.services.llm_service import get_llm_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Initializing database...")
        await init_db()
        logger.info("Database initialized successfully")
        
        logger.info("Initializing OCR service...")
        try:
            ocr_service = get_ocr_service()
            logger.info("OCR service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR service: {e}", exc_info=True)
            logger.warning("Continuing without OCR service - stage3 will fail")
        
        logger.info("Initializing stamp detection service...")
        try:
            stamp_service = get_stamp_service()
            logger.info("Stamp detection service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize stamp detection service: {e}", exc_info=True)
            logger.warning("Continuing without stamp detection service - stage4 will fail")
        
        logger.info("Initializing LLM service...")
        try:
            llm_service = get_llm_service()
            logger.info("LLM service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}", exc_info=True)
            logger.warning("Continuing without LLM service - stage5 will fail")
        
        logger.info("Registering processors...")
        queue_service.register_processor(ProcessorType.DEFAULT.value, [
            Stage1ChecksumProcessor(),
            Stage2SHA1Processor(),
            Stage3OCRProcessor(),
            Stage4StampProcessor(),
            # Stage5DataExtractionProcessor() llm is shit
        ])
        logger.info("Processors registered successfully")
        
        logger.info("Starting queue workers...")
        db_factory = get_database_factory()
        await queue_service.start_workers(db_factory.get_session_factory())
        logger.info("Queue workers started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down queue workers...")
        await queue_service.stop_workers()
        logger.info("Application shutdown complete")


app = FastAPI(
    title="Object Processing API",
    description="Enterprise-grade object processing system with async queue management",
    version="1.0.0",
    lifespan=lifespan
)
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Object Processing API",
        "version": "1.0.0",
        "status": "operational"
    }
