from typing import Dict, Any, Optional
import asyncio


class ProcessingContext:
    _contexts: Dict[str, Dict[str, Any]] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def get(cls, job_id: str, key: Optional[str] = None, default: Any = None) -> Any:
        async with cls._lock:
            context = cls._contexts.get(job_id, {})
            if key is None:
                return context
            return context.get(key, default)

    @classmethod
    async def set(cls, job_id: str, key: str, value: Any):
        async with cls._lock:
            if job_id not in cls._contexts:
                cls._contexts[job_id] = {}
            cls._contexts[job_id][key] = value

    @classmethod
    async def get_all(cls, job_id: str) -> Dict[str, Any]:
        async with cls._lock:
            return cls._contexts.get(job_id, {}).copy()

    @classmethod
    async def clear(cls, job_id: str):
        async with cls._lock:
            if job_id in cls._contexts:
                del cls._contexts[job_id]
