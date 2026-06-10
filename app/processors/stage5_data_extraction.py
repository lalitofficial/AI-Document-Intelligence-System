from typing import Dict, Any, List
from app.processors.base import BaseProcessor
from app.enums import StageName
from app.services.llm_service import get_llm_service
import asyncio
import json


class Stage5DataExtractionProcessor(BaseProcessor):
    def __init__(self):
        super().__init__(StageName.STAGE5_DATA_EXTRACTION, use_process=False)

    def process_sync(self, job_id: str, params: List[str], xcom_data: Dict[str, Any]) -> Dict[str, Any]:
        # Get OCR output from stage3
        stage3_output = xcom_data.get("stage3_ocr_output")
        if not stage3_output:
            raise ValueError("Stage3 OCR output not found in xcom context")
        
        # Get stamp detection from stage4
        stage4_output = xcom_data.get("stage4_stamp_output")
        if not stage4_output:
            raise ValueError("Stage4 stamp detection output not found in xcom context")
        
        # Extract OCR text
        ocr_text = ""
        if isinstance(stage3_output, str):
            try:
                stage3_data = json.loads(stage3_output.replace("'", '"'))
            except:
                stage3_data = {"raw_output": stage3_output}
        else:
            stage3_data = stage3_output
        
        if isinstance(stage3_data, dict):
            raw_output = stage3_data.get("raw_output", "")
            if isinstance(raw_output, str):
                ocr_text = raw_output
            elif isinstance(raw_output, dict):
                ocr_text = json.dumps(raw_output)
        else:
            ocr_text = str(stage3_data)
        # Extract stamp detection data
        stamp_detection = {}
        if isinstance(stage4_output, str):
            try:
                stamp_detection = json.loads(stage4_output.replace("'", '"'))
            except:
                stamp_detection = {"status": "unknown"}
        else:
            stamp_detection = stage4_output
        
        llm_service = get_llm_service()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                llm_service.extract_structured_data(ocr_text, stamp_detection)
            )
            return result
        finally:
            loop.close()
