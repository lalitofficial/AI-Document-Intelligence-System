class ObjectProcessingException(Exception):
    """Base exception for object processing errors"""
    pass


class JobNotFoundException(ObjectProcessingException):
    """Raised when a job is not found"""
    pass


class StageProcessingException(ObjectProcessingException):
    """Raised when a stage processing fails"""
    pass


class QueueFullException(ObjectProcessingException):
    """Raised when the queue is full"""
    pass


class DatabaseException(ObjectProcessingException):
    """Raised when database operations fail"""
    pass
