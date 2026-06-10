from app.core.config import settings
from app.core.logger import get_logger
from app.enums.app_config import OCRProvider
from .ocr_service_donut import OCRServiceInterface, get_donut_ocr_service
from .ocr_service_thirdparty import get_thirdparty_ocr_service
from .ocr_service_local import get_local_ocr_service

logger = get_logger(__name__)

__all__ = ["OCRServiceInterface", "get_ocr_service"]


def get_ocr_service() -> OCRServiceInterface:
    """
    Factory function that returns the appropriate OCR service based on config.
    Config option: ocr_provider (OCRProvider enum)
    - OCRProvider.LOCAL: PaddleOCR with Tesseract fallback (CPU-only, free)
    - OCRProvider.DONUT: Local Donut transformer model
    - OCRProvider.THIRDPARTY: OCR.space API
    """
    provider = getattr(settings, "ocr_provider", OCRProvider.LOCAL)

    if provider == OCRProvider.THIRDPARTY:
        logger.info("Using ThirdParty OCR service (OCR.space)")
        return get_thirdparty_ocr_service()
    if provider == OCRProvider.DONUT:
        logger.info("Using Donut OCR service (local model)")
        return get_donut_ocr_service()
    logger.info("Using Local OCR service (PaddleOCR/Tesseract)")
    return get_local_ocr_service()
