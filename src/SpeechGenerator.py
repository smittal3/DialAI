import threading
import queue
import asyncio
import json
import time
import boto3
from config import AppConfig, api_request_list
from typing import Optional
from botocore.config import Config
from Logger import Logger, LogComponent
from Metrics import Metrics, MetricType


class SpeechGenerator:
    def __init__(self, config, user_interrupt, bedrock_complete, speech_complete, bedrock_to_stt, output_stream):
        self.config = config
        self.user_interrupt = user_interrupt
        self.bedrock_complete = bedrock_complete
        self.speech_complete = speech_complete
        self.bedrock_to_stt = bedrock_to_stt
        self.output_stream = output_stream
        self.thread: Optional[threading.Thread] = None
        self.is_running = True
        self.logger = Logger()
        self.metrics = Metrics()
        self.logger.info(LogComponent.SPEECH, f"Initializing Polly client for region {config.aws_region}")
        self.polly = boto3.client('polly', region_name=config.aws_region)
        self.chunk_size = 640

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._generate_audio)
        self.metrics.start_metric(MetricType.THREAD_LIFETIME, "speech_thread")
        self.thread.start()
        self.logger.info(LogComponent.SPEECH, "Speech generation thread started")

    def stop(self):
        if self.thread is not None:
            self.logger.info(LogComponent.SPEECH, "Stopping speech generator thread")
            self.is_running = False
            self.thread.join()
            self.metrics.end_metric(MetricType.THREAD_LIFETIME, "speech_thread")
            self.thread = None

    def _generate_audio(self):
        first_chunk = True
        while self.is_running:
            try:
                if self.user_interrupt.is_set():
                    self.logger.info(LogComponent.SPEECH, "User interrupt detected, clearing output stream")
                    self.speech_complete.set()
                    with self.output_stream.mutex:
                        self.output_stream.queue.clear()
                    break

                text = self.bedrock_to_stt.get(timeout=0.01)
                if text:
                    if first_chunk:
                        self.metrics.end_metric(MetricType.SPEECH_PROCESSING, "first_chunk_bedrock_to_stt")

                    self.logger.debug(LogComponent.SPEECH, f"Generating speech for text: {text}")
                    self.metrics.start_metric(MetricType.API_LATENCY, "polly_synthesis")
                    response = self.polly.synthesize_speech(
                        Text=text,
                        Engine=self.config.polly['engine'],
                        LanguageCode=self.config.polly['language'],
                        VoiceId=self.config.polly['voice'],
                        OutputFormat=self.config.polly['outputFormat']
                    )

                    if first_chunk:
                        self.metrics.record_time_to_first_token("polly_synthesis", "polly_first_token")
                    self.metrics.end_metric(MetricType.API_LATENCY, "polly_synthesis")
                    
                    stream = response['AudioStream']
                    while True:
                        audio_chunk = stream.read(self.chunk_size)
                        if not audio_chunk or self.user_interrupt.is_set():
                            break
                        if first_chunk:
                            self.metrics.start_metric(MetricType.SPEECH_PROCESSING, "first_chunk_to_output_stream")
                            first_chunk = False
                        self.output_stream.put(audio_chunk)
                       
            except queue.Empty:    
                if self.bedrock_complete.is_set():
                    self.logger.info(LogComponent.SPEECH, "Bedrock complete, finishing speech generation")
                    self.speech_complete.set()
                    break
                continue
            except Exception as e:
                self.logger.error(LogComponent.SPEECH, f"Error in Polly processing: {e}")
