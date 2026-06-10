from typing import Dict, Any, List
from app.processors.base import BaseProcessor
from app.enums import StageName
import hashlib


class Stage2SHA1Processor(BaseProcessor):
    def __init__(self):
        super().__init__(StageName.STAGE2_SHA1, use_process=False)

    def process_sync(self, job_id: str, params: List[str], xcom_data: Dict[str, Any]) -> Dict[str, Any]:
        stage1_output = xcom_data.get("stage1_checksum_output")
        if not stage1_output:
            raise ValueError("Stage1 output not found in xcom context")
        
        object_data = xcom_data.get("object_data")
        if not object_data:
            raise ValueError("Object data not found in xcom context")
        
        if isinstance(object_data, str):
            object_bytes = object_data.encode('latin-1')
        elif isinstance(object_data, bytes):
            object_bytes = object_data
        else:
            object_bytes = bytes(object_data)
        
        sha1_hash = hashlib.sha1(object_bytes).hexdigest()
        checksum = stage1_output.get("checksum", "")
        
        return {
            "sha1_hash": sha1_hash,
            "checksum": checksum,
            "algorithm": "sha1"
        }
