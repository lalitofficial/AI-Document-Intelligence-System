from typing import Dict, Any, List
from app.processors.base import BaseProcessor
from app.enums import StageName
from app.services.ocr_service import get_ocr_service
import asyncio


class Stage3OCRProcessor(BaseProcessor):
    def __init__(self):
        super().__init__(StageName.STAGE3_OCR, use_process=False)

    def process_sync(self, job_id: str, params: List[str], xcom_data: Dict[str, Any]) -> Dict[str, Any]:
        object_data = xcom_data.get("object_data")
        if not object_data:
            raise ValueError("Object data not found in xcom context")

        if isinstance(object_data, str):
            object_bytes = object_data.encode('latin-1')
        elif isinstance(object_data, bytes):
            object_bytes = object_data
        else:
            object_bytes = bytes(object_data)
        
        task_prompt = params[0] if params and len(params) > 0 else "<s_receipt>"
        
        ocr_service = get_ocr_service()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ocr_service.extract_text(object_bytes, task_prompt)
            )
            return result
        finally:
            loop.close()
