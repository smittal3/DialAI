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
        self.logger.info(LogComponent.SPEECH, f"Initializing Polly client for region {config.aws_region}")
        self.polly = boto3.client('polly', region_name=config.aws_region)
        self.chunk_size = 640

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._generate_audio)
        self.thread.start()
        self.logger.info(LogComponent.SPEECH, "Speech generation thread started")

    def stop(self):
        if self.thread is not None:
            self.logger.info(LogComponent.SPEECH, "Stopping speech generator thread")
            self.is_running = False
            self.thread.join()
            self.thread = None

    def _generate_audio(self):
        while self.is_running:
            try:
                if self.user_interrupt.is_set():
                    self.logger.info(LogComponent.SPEECH, "User interrupt detected, clearing output stream")
                    self.speech_complete.set()
                    with self.output_stream.mutex:
                        self.output_stream.queue.clear()
                    break

                text = self.bedrock_to_stt.get(timeout=0.1)
                if text:
                    self.logger.debug(LogComponent.SPEECH, f"Generating speech for text: {text}")
                    start_time = time.time()
                    response = self.polly.synthesize_speech(
                        Text=text,
                        Engine=self.config.polly['engine'],
                        LanguageCode=self.config.polly['language'],
                        VoiceId=self.config.polly['voice'],
                        OutputFormat=self.config.polly['outputFormat']
                    )
                    overall_time = time.time() - start_time
                    self.logger.debug(LogComponent.SPEECH, f"Polly response received in {overall_time:.2f} seconds")
                    
                    stream = response['AudioStream']
                    while True:
                        audio_chunk = stream.read(self.chunk_size)
                        if not audio_chunk or self.user_interrupt.is_set():
                            break

                        self.output_stream.put(audio_chunk)
                       
            except queue.Empty:    
                if self.bedrock_complete.is_set():
                    self.logger.info(LogComponent.SPEECH, "Bedrock complete, finishing speech generation")
                    self.speech_complete.set()
                    break
                continue
            except Exception as e:
                self.logger.error(LogComponent.SPEECH, f"Error in Polly processing: {e}")
