from typing import Dict, Any, List
from app.processors.base import BaseProcessor
from app.enums import StageName
import hashlib


class Stage1ChecksumProcessor(BaseProcessor):
    def __init__(self):
        super().__init__(StageName.STAGE1_CHECKSUM, use_process=False)

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
        
        checksum = hashlib.md5(object_bytes).hexdigest()
        
        return {"checksum": checksum, "algorithm": "md5"}
