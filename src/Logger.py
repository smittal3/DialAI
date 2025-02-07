import logging
import os
from datetime import datetime
from typing import Dict, Optional
import threading
from enum import Enum

class LogComponent(Enum):
    VAD = "\033[94m"  # Blue
    TRANSCRIBE = "\033[92m"  # Green
    INFERENCE = "\033[95m"  # Purple
    SPEECH = "\033[93m"  # Yellow
    WEBSOCKET = "\033[96m"  # Cyan
    ORCHESTRATOR = "\033[91m"  # Red
    SYSTEM = "\033[97m"  # White
    DATABASE = "\033[90m"  # Gray

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'component'):
            record.msg = f"{record.component.value} {record.msg}\033[0m"
        return super().format(record)

class PlainFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'component'):
            record.msg = f"[{record.component.name}] {record.msg}"
        return super().format(record)

class Logger:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not Logger._initialized:
            with Logger._lock:
                if not Logger._initialized:
                    self._setup_logger()
                    Logger._initialized = True

    def _setup_logger(self):
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/dialai_{timestamp}.log'

        # Configure the logger
        self.logger = logging.getLogger('DialAI')
        self.logger.setLevel(logging.DEBUG)

        # File handler - without colors
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = PlainFormatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter('%(message)s')
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _log(self, level: int, component: LogComponent, message: str):
        record = logging.LogRecord(
            'DialAI', level, '', 0, message, (), None
        )
        record.component = component
        self.logger.handle(record)

    def info(self, component: LogComponent, message: str):
        self._log(logging.INFO, component, message)

    def debug(self, component: LogComponent, message: str):
        self._log(logging.DEBUG, component, message)

    def error(self, component: LogComponent, message: str):
        self._log(logging.ERROR, component, message)

    def warning(self, component: LogComponent, message: str):
        self._log(logging.WARNING, component, message) 