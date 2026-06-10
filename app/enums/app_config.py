from enum import Enum


class ProcessorType(str, Enum):
    DEFAULT = "default"
    CUSTOM = "custom"


class StageName(str, Enum):
    STAGE1_CHECKSUM = "stage1_checksum"
    STAGE2_SHA1 = "stage2_sha1"
    STAGE3_OCR = "stage3_ocr"
    STAGE4_STAMP = "stage4_stamp"
    STAGE5_DATA_EXTRACTION = "stage5_data_extraction"


class OCRProvider(str, Enum):
    DONUT = "donut"
    THIRDPARTY = "thirdparty"
    LOCAL = "local"


class StampServiceType(str, Enum):
    YOLO_TRANSFORMER = "yolo_transformer"
    IMAGE_PROCESSING = "image_processing"
    MOBILENET = "mobilenet"