from abc import ABC, abstractmethod
from typing import Dict, Any, List
from app.core.context import ProcessingContext
from app.enums import ProcessorStatus, StageName
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools


class XCom:
    @staticmethod
    async def get(job_id: str, key: str, default: Any = None) -> Any:
        return await ProcessingContext.get(job_id, key, default)
    
    @staticmethod
    async def set(job_id: str, key: str, value: Any):
        await ProcessingContext.set(job_id, key, value)
    
    @staticmethod
    async def get_all(job_id: str) -> Dict[str, Any]:
        return await ProcessingContext.get_all(job_id)


class BaseProcessor(ABC):
    _thread_pool = ThreadPoolExecutor(max_workers=4)
    _process_pool = ProcessPoolExecutor(max_workers=2)

    def __init__(self, stage_name: StageName, use_process: bool = False):
        self.stage_name = stage_name.value if isinstance(stage_name, StageName) else stage_name
        self.use_process = use_process

    @abstractmethod
    def process_sync(self, job_id: str, params: List[str], xcom_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    async def process(self, job_id: str, params: List[str], context_data: Dict[str, Any]) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        executor = self._process_pool if self.use_process else self._thread_pool
        
        process_func = functools.partial(
            self.process_sync,
            job_id=job_id,
            params=params,
            xcom_data=context_data
        )
        
        result = await loop.run_in_executor(executor, process_func)
        return result

    async def execute(self, job_id: str, params: List[str]) -> Dict[str, Any]:
        context_data = await ProcessingContext.get_all(job_id)
        try:
            result = await self.process(job_id, params, context_data)
            await ProcessingContext.set(job_id, f"{self.stage_name}_output", result)
            return {
                "status": ProcessorStatus.SUCCESS.value,
                "output": result,
                "error": None,
                "stack_trace": None
            }
        except Exception as e:
            import traceback
            stack_trace = traceback.format_exc()
            return {
                "status": ProcessorStatus.ERROR.value,
                "output": None,
                "error": str(e),
                "stack_trace": stack_trace
            }
