import time
import threading
from typing import Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from Logger import Logger, LogComponent

class MetricType(Enum):
    THREAD_LIFETIME = "thread_lifetime"
    API_LATENCY = "api_latency"
    TIME_TO_FIRST_TOKEN = "time_to_first_token"  # Specific metric for streaming APIs
    SPEECH_PROCESSING = "speech_processing"
    TRANSCRIPTION = "transcription"
    INFERENCE = "inference"
    VOICE_DETECTION = "voice_detection"

@dataclass
class MetricEntry:
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None

class Metrics:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Metrics, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not Metrics._initialized:
            with Metrics._lock:
                if not Metrics._initialized:
                    self._metrics: Dict[MetricType, Dict[str, MetricEntry]] = {
                        metric_type: {} for metric_type in MetricType
                    }
                    self.logger = Logger()
                    Metrics._initialized = True

    def start_metric(self, metric_type: MetricType, identifier: str) -> None:
        """Start timing a metric."""
        with self._lock:
            self._metrics[metric_type][identifier] = MetricEntry(start_time=time.time())
            self.logger.debug(LogComponent.SYSTEM, f"Started {metric_type.value} metric for {identifier}")

    def record_time_to_first_token(self, start_identifier: str, identifier: str) -> float:
        """Record time to first token for streaming APIs."""
        with self._lock:
            if start_identifier in self._metrics[MetricType.API_LATENCY]:
                start_entry = self._metrics[MetricType.API_LATENCY][start_identifier]
                current_time = time.time()
                time_to_first = current_time - start_entry.start_time
                
                # Record as a separate metric
                self._metrics[MetricType.TIME_TO_FIRST_TOKEN][identifier] = MetricEntry(
                    start_time=start_entry.start_time,
                    end_time=current_time,
                    duration=time_to_first
                )
                
                self.logger.info(
                    LogComponent.SYSTEM,
                    f"Time to first token for {identifier}: {time_to_first:.3f}s"
                )
                return time_to_first
            return 0.0

    def end_metric(self, metric_type: MetricType, identifier: str) -> float:
        """End timing a metric and return the duration."""
        with self._lock:
            if identifier not in self._metrics[metric_type]:
                self.logger.warning(LogComponent.SYSTEM, f"No start time found for {metric_type.value} metric {identifier}")
                return 0.0

            entry = self._metrics[metric_type][identifier]
            entry.end_time = time.time()
            entry.duration = entry.end_time - entry.start_time
            
            self.logger.info(
                LogComponent.SYSTEM, 
                f"{metric_type.value} metric for {identifier}: {entry.duration:.3f}s"
            )
            return entry.duration

    def get_metric(self, metric_type: MetricType, identifier: str) -> Optional[float]:
        """Get the duration of a specific metric."""
        with self._lock:
            entry = self._metrics[metric_type].get(identifier)
            return entry.duration if entry and entry.duration is not None else None

    def get_average_metric(self, metric_type: MetricType) -> Optional[float]:
        """Get the average duration for a metric type."""
        with self._lock:
            durations = [
                entry.duration 
                for entry in self._metrics[metric_type].values() 
                if entry.duration is not None
            ]
            return sum(durations) / len(durations) if durations else None

    def clear_metrics(self, metric_type: Optional[MetricType] = None) -> None:
        """Clear metrics for a specific type or all metrics if type not specified."""
        with self._lock:
            if metric_type:
                self._metrics[metric_type].clear()
            else:
                for metric_dict in self._metrics.values():
                    metric_dict.clear()

    def get_streaming_metrics_summary(self) -> None:
        """Get a summary of streaming metrics for all services."""
        with self._lock:
            # Get all first token metrics
            first_token_metrics = {
                identifier: entry.duration
                for identifier, entry in self._metrics[MetricType.TIME_TO_FIRST_TOKEN].items()
                if entry.duration is not None
            }
            
            # Group by service
            service_metrics = {
                'transcribe': [],
                'bedrock': [],
                'polly': []
            }
            
            for identifier, duration in first_token_metrics.items():
                if 'transcribe' in identifier:
                    service_metrics['transcribe'].append(duration)
                elif 'bedrock' in identifier:
                    service_metrics['bedrock'].append(duration)
                elif 'polly' in identifier:
                    service_metrics['polly'].append(duration)
            
            # Log summary for each service
            for service, durations in service_metrics.items():
                if durations:
                    avg = sum(durations) / len(durations)
                    min_time = min(durations)
                    max_time = max(durations)
                    self.logger.info(
                        LogComponent.SYSTEM,
                        f"{service} streaming metrics summary:\n"
                        f"  Average time to first token: {avg:.3f}s\n"
                        f"  Best time to first token: {min_time:.3f}s\n"
                        f"  Worst time to first token: {max_time:.3f}s\n"
                        f"  Sample size: {len(durations)}"
                    )

    def get_service_first_token_metrics(self, service: str) -> Tuple[Optional[float], Optional[float]]:
        """Get the latest and average time to first token for a specific service.
        
        Args:
            service: The service name ('transcribe', 'bedrock', or 'polly')
            
        Returns:
            Tuple of (latest_time, average_time)
        """
        with self._lock:
            service_metrics = [
                entry.duration
                for identifier, entry in self._metrics[MetricType.TIME_TO_FIRST_TOKEN].items()
                if service in identifier and entry.duration is not None
            ]
            
            if not service_metrics:
                return None, None
                
            latest = service_metrics[-1]  # Last recorded metric
            average = sum(service_metrics) / len(service_metrics)
            return latest, average 