import threading
from typing import Optional
from Logger import Logger, LogComponent
from Metrics import Metrics

class BaseThread:
    def __init__(self, name: str):
        self.thread: Optional[threading.Thread] = None
        self.is_running = True
        self.logger = Logger()
        self.metrics = Metrics()
        self.name = name
        
    def start(self):
        if self.thread is not None and self.thread.is_alive():
            self.logger.warning(LogComponent.SYSTEM, f"{self.name} thread is already running")
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._run_process, name=self.name)
        self.thread.daemon = False  # Make thread daemon by default so it exits when main program exits
        self.thread.start()
        self.logger.info(LogComponent.SYSTEM, f"{self.name} thread started")
        
    def stop(self):
        if self.thread is not None:
            self.logger.info(LogComponent.SYSTEM, f"Stopping {self.name} thread")
            self.is_running = False
            self.thread.join(timeout=1)
            self.thread = None
            
    def _run_process(self):
        """Override this method in derived classes.
        
        Implementation should periodically check self.is_running and exit when False.
        Example:
            while self.is_running:
                # do work
                time.sleep(0.1)  # Prevent tight loops
        """
        raise NotImplementedError 