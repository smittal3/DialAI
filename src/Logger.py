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

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _format_message(self, component: LogComponent, message: str) -> str:
        return f"{component.value}[{component.name}] {message}\033[0m"

    def info(self, component: LogComponent, message: str):
        formatted_message = self._format_message(component, message)
        self.logger.info(formatted_message)

    def debug(self, component: LogComponent, message: str):
        formatted_message = self._format_message(component, message)
        self.logger.debug(formatted_message)

    def error(self, component: LogComponent, message: str):
        formatted_message = self._format_message(component, message)
        self.logger.error(formatted_message)

    def warning(self, component: LogComponent, message: str):
        formatted_message = self._format_message(component, message)
        self.logger.warning(formatted_message) 