from typing import Any, Dict, Optional
import asyncio
import io

from PIL import Image

from app.core.logger import get_logger
from .ocr_service_thirdparty import OCRServiceInterface

logger = get_logger(__name__)

# Both engines are optional so the API can boot on machines that only have
# one of them; the factory falls back PaddleOCR -> Tesseract at call time.
try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - depends on local install
    PaddleOCR = None

try:
    import pytesseract
except Exception:  # pragma: no cover - depends on local install
    pytesseract = None


class LocalOCRService(OCRServiceInterface):
    """
    CPU-only OCR with no per-call API cost: PaddleOCR (multilingual,
    angle-classified) when installed, falling back to Tesseract.
    """

    _instance: Optional["LocalOCRService"] = None
    _paddle: Optional[Any] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            if PaddleOCR is not None:
                try:
                    self._paddle = PaddleOCR(use_angle_cls=True, lang="en")
                    logger.info("LocalOCRService using PaddleOCR")
                except Exception as exc:  # pragma: no cover
                    logger.warning("PaddleOCR unavailable (%s)", exc)
            if self._paddle is None and pytesseract is None:
                raise RuntimeError(
                    "Local OCR requires paddleocr or pytesseract; install one "
                    "or set OCR_PROVIDER=thirdparty/donut."
                )
            if self._paddle is None:
                logger.info("LocalOCRService using Tesseract")
            LocalOCRService._initialized = True

    def _extract_sync(self, image_data: bytes) -> Dict[str, Any]:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        if self._paddle is not None:
            import numpy as np

            result = self._paddle.ocr(np.array(image), cls=True)
            lines = []
            confidences = []
            for page in result or []:
                for entry in page or []:
                    text, confidence = entry[1][0], float(entry[1][1])
                    lines.append(text)
                    confidences.append(confidence)
            return {
                "raw_output": "\n".join(lines),
                "engine": "paddleocr",
                "confidence": (
                    sum(confidences) / len(confidences) if confidences else None
                ),
            }

        text = pytesseract.image_to_string(image)
        return {"raw_output": text, "engine": "tesseract", "confidence": None}

    async def extract_text(
        self,
        image_data: bytes,
        task_prompt: str = "<s_receipt>",  # kept for interface compatibility
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self._extract_sync, image_data)


def get_local_ocr_service() -> LocalOCRService:
    return LocalOCRService()
