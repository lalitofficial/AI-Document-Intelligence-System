from typing import Dict, Any, List
from app.processors.base import BaseProcessor
from app.enums import StageName
from app.services.stamp_service import get_stamp_service
import asyncio


class Stage4StampProcessor(BaseProcessor):
    def __init__(self):
        super().__init__(StageName.STAGE4_STAMP, use_process=False)

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
        
        stamp_service = get_stamp_service()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                stamp_service.detect_marks(object_bytes)
            )
            
            # Transform result to expected format for backward compatibility
            if result.get("status") == "success" and result.get("result"):
                marks = result["result"]
                return {
                    "status": "success",
                    "has_stamp": marks.get("stamp", {}).get("present", False),
                    "stamp_confidence": marks.get("stamp", {}).get("confidence", 0.0),
                    "has_signature": marks.get("signature", {}).get("present", False),
                    "signature_confidence": marks.get("signature", {}).get("confidence", 0.0),
                    "stamp": marks.get("stamp", {}),
                    "signature": marks.get("signature", {}),
                    "device_used": result.get("device_used", "unknown")
                }
            else:
                return {
                    "status": "error",
                    "error": result.get("error", "Unknown error"),
                    "has_stamp": False,
                    "stamp_confidence": 0.0,
                    "has_signature": False,
                    "signature_confidence": 0.0
                }
        finally:
            loop.close()
