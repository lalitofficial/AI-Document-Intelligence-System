from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv
import os
from pathlib import Path
from app.enums.app_config import OCRProvider, StampServiceType


class Settings(BaseSettings):
    max_workers: int = 2
    max_concurrent_jobs: int = 3
    database_url: Optional[str] = None
    bucket_storage_folder: str = "./bucket_storage"
    ocr_provider: OCRProvider = OCRProvider.LOCAL
    ocr_api_url: str = "https://api.ocr.space/parse/image"
    ocr_api_key: Optional[str] = None
    ocr_force_cpu: bool = False
    ocr_device: Optional[str] = "cpu"
    huggingface_token: Optional[str] = None
    stamp_model_path: Optional[str] = None
    stamp_force_cpu: bool = False
    stamp_device: Optional[str] = "cpu"
    use_image_processing_stamp: bool = True
    stamp_service_type: StampServiceType = StampServiceType.YOLO_TRANSFORMER
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = "deepseek/deepseek-r1-0528:free"
    openrouter_base_url: Optional[str] = "https://openrouter.ai/api/v1"
    openrouter_site_url: Optional[str] = None
    openrouter_site_name: Optional[str] = "Loan Invoice Document AI"

    class Config:
        env_file = ".env"

    def __init__(self):
        super().__init__()
        load_dotenv()
        self._ensure_bucket_storage_folder()

    def _ensure_bucket_storage_folder(self):
        storage_path = Path(self.bucket_storage_folder)
        storage_path.mkdir(parents=True, exist_ok=True)


settings = Settings()
