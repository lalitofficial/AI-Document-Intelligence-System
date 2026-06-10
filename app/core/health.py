from typing import Dict, Any
from app.db import get_database_factory
from app.services.queue_service import queue_service
from app.core.logger import get_logger

logger = get_logger(__name__)


async def check_database_health() -> Dict[str, Any]:
    try:
        from sqlalchemy import text
        factory = get_database_factory()
        async with factory.get_session() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}


async def check_queue_health() -> Dict[str, Any]:
    try:
        queue_size = queue_service.queue.qsize()
        workers_running = len([w for w in queue_service.workers if not w.done()])
        return {
            "status": "healthy",
            "queue_size": queue_size,
            "workers_running": workers_running,
            "max_workers": len(queue_service.workers)
        }
    except Exception as e:
        logger.error(f"Queue health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


async def get_health_status() -> Dict[str, Any]:
    db_health = await check_database_health()
    queue_health = await check_queue_health()
    
    overall_status = "healthy" if (
        db_health.get("status") == "healthy" and
        queue_health.get("status") == "healthy"
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "database": db_health,
        "queue": queue_health
    }
