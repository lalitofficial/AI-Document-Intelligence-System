from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PIL import Image
import io
import asyncio
import aiohttp
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class OCRServiceInterface(ABC):
    @abstractmethod
    async def extract_text(
        self,
        image_data: bytes,
        task_prompt: str = "<s_receipt>"  # kept for compatibility
    ) -> Dict[str, Any]:
        pass


class ThirdPartyOCRService(OCRServiceInterface):
    """
    OCR.space based OCR service
    Async, CPU-safe, production stable
    """

    _instance: Optional["ThirdPartyOCRService"] = None
    _initialized: bool = False

    TIMEOUT_SECONDS = 60
    POLL_INTERVAL = 2

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ThirdPartyOCRService._initialized:
            self.ocr_api_url = getattr(settings, "ocr_api_url", "https://api.ocr.space/parse/image")
            self.ocr_api_key = getattr(settings, "ocr_api_key", None)
            if not self.ocr_api_key:
                raise ValueError("ocr_api_key must be set in config")
            ThirdPartyOCRService._initialized = True

    async def extract_text(
        self,
        image_data: bytes,
        task_prompt: str = "<s_receipt>"
    ) -> Dict[str, Any]:
        try:
            image_bytes = self._normalize_image(image_data)

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.TIMEOUT_SECONDS)
            ) as session:
                response = await self._submit_ocr_request(session, image_bytes)
                parsed = self._parse_ocr_response(response)

            logger.info("[OCR] OCR.space extraction successful")

            return {
                "status": "success",
                "raw_output": parsed,
                "device_used": "api",
                "provider": "ocr.space"
            }

        except asyncio.TimeoutError:
            logger.error("OCR.space request timed out")
            return {
                "status": "error",
                "error": "OCR request timed out",
                "raw_output": None
            }

        except Exception as e:
            logger.error("OCR.space extraction failed", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "raw_output": None
            }

    async def _submit_ocr_request(
        self,
        session: aiohttp.ClientSession,
        image_bytes: bytes
    ) -> Dict[str, Any]:
        """
        Submits OCR request and waits for response
        """

        form = aiohttp.FormData()
        form.add_field(
            "file",
            image_bytes,
            filename="image.jpg",
            content_type="image/jpeg"
        )

        form.add_field("language", "eng")
        form.add_field("isOverlayRequired", "false")
        form.add_field("OCREngine", "2")  # Best engine
        form.add_field("scale", "true")

        headers = {
            "apikey": self.ocr_api_key
        }

        async with session.post(
            self.ocr_api_url,
            data=form,
            headers=headers
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"OCR.space HTTP {resp.status}")
            return await resp.json()

    @staticmethod
    def _parse_ocr_response(response: Dict[str, Any]) -> Any:
        """
        Normalize OCR.space response
        """
        if response.get("IsErroredOnProcessing"):
            error_message = response.get("ErrorMessage", "Unknown OCR error")
            raise RuntimeError(error_message)

        parsed_results = response.get("ParsedResults")
        if not parsed_results:
            return ""

        extracted_text = []
        for block in parsed_results:
            text = block.get("ParsedText", "")
            if text:
                extracted_text.append(text.strip())

        return "\n".join(extracted_text)

    @staticmethod
    def _normalize_image(image_data: bytes) -> bytes:
        """
        Ensures image is RGB JPG (OCR.space works best with JPG)
        """
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        max_size = 2000
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90)
        return output.getvalue()


def get_thirdparty_ocr_service() -> OCRServiceInterface:
    return ThirdPartyOCRService()
