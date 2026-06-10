import logging
import sys
from typing import Optional


class Logger:
    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls, name: str = "object_processing") -> logging.Logger:
        if cls._instance is None:
            cls._instance = logging.getLogger(name)
            cls._instance.setLevel(logging.INFO)
            
            if not cls._instance.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setLevel(logging.INFO)
                
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(formatter)
                cls._instance.addHandler(handler)
        
        return cls._instance


def get_logger(name: str = "object_processing") -> logging.Logger:
    return Logger.get_logger(name)
