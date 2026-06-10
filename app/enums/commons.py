from enum import Enum


class JobStatus(str, Enum):
    IN_QUEUE = "inqueue"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


class StageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"
    FAILED = "failed"


class ProcessorStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
