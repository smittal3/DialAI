import threading
import queue
import asyncio
from config import AppConfig
from typing import Optional
from botocore.exceptions import BotoCoreError, ClientError
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from Logger import Logger, LogComponent
from Metrics import Metrics, MetricType
from BaseThread import BaseThread
import time

class TranscribeHandler(TranscriptResultStreamHandler):
    def __init__(self,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.transcript_buffer = []
        self.metrics = Metrics()
        self.first_chunk = True
        self.second_chunk = False

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        if self.first_chunk:
            self.metrics.record_time_to_first_token("transcribe_stream", "transcribe_first_token")
            self.first_chunk = False
            self.second_chunk = True
            self.metrics.start_metric(MetricType.SPEECH_PROCESSING, "first_to_second_chunk_transcribe_stream")
        if self.second_chunk:
            self.metrics.end_metric(MetricType.SPEECH_PROCESSING, "first_to_second_chunk_transcribe_stream")
            self.second_chunk = False

        results = transcript_event.transcript.results
        for result in results:
            if not result.is_partial:
                transcript = result.alternatives[0].transcript
                self.transcript_buffer.append(transcript)

    def get_transcript(self):
        transcript = ' '.join(self.transcript_buffer)
        self.transcript_buffer = []
        return transcript


# Blocks when silent, otherwise triggers transcribe and puts result in output queue to bedrock
class Transcribe(BaseThread):
    def __init__(self, 
                 vad_to_transcribe: queue.Queue,
                 transcribe_to_bedrock: queue.Queue,
                 user_interrupt: threading.Event,
                 silence_indicator: threading.Event,
                 system_interrupt: threading.Event,
                 config: AppConfig):
        super().__init__(name="Transcribe")
        self.vad_to_transcribe = vad_to_transcribe
        self.transcribe_to_bedrock = transcribe_to_bedrock
        self.config = config
        self.user_interrupt = user_interrupt
        self.silence_indicator = silence_indicator
        self.system_interrupt = system_interrupt
        
        # Initialize AWS client
        self.logger.info(LogComponent.TRANSCRIBE, f"Initializing AWS Transcribe client for region {config.aws_region}")
        self.client = TranscribeStreamingClient(region=config.aws_region)

    def _run_process(self):
        asyncio.run(self._transcribe_stream())

    async def _transcribe_stream(self):
        while self.is_running:
            try:
                self.logger.debug(LogComponent.TRANSCRIBE, "Waiting for silence indicator to clear")

                while self.silence_indicator.is_set():
                    if self.system_interrupt.is_set():
                        break
                    await asyncio.sleep(0.1)

                if self.system_interrupt.is_set():
                    break

                    
                self.logger.info(LogComponent.TRANSCRIBE, "Starting AWS stream transcription")
                self.metrics.start_metric(MetricType.TRANSCRIPTION, "initializing_transcribe_stream")
                stream = await self.client.start_stream_transcription(
                    language_code="en-US",
                    media_sample_rate_hz=self.config.sample_rate,
                    media_encoding="pcm"
                )

                self.metrics.end_metric(MetricType.TRANSCRIPTION, "initializing_transcribe_stream")
                
                handler = TranscribeHandler(stream.output_stream)
                stream.handler = handler

                async def send_audio_events():
                    first_chunk = True
                    while True:
                        try:    
                            data = self.vad_to_transcribe.get(timeout=0.01)
                            if first_chunk:
                                self.metrics.start_metric(MetricType.API_LATENCY, "transcribe_stream")
                                self.metrics.end_metric(MetricType.TRANSCRIPTION, "first_chunk_vad_to_transcribe")
                                first_chunk = False
                            await stream.input_stream.send_audio_event(audio_chunk=data.tobytes())
                            if self.system_interrupt.is_set():
                                break
                        except queue.Empty:
                            if self.silence_indicator.is_set():
                                self.logger.debug(LogComponent.TRANSCRIBE, "Silence detected, ending transcription stream")
                                break
                            continue
                        except Exception as e:
                            self.logger.error(LogComponent.TRANSCRIBE, f"Error in audio stream: {e}")
                            break

                    await stream.input_stream.end_stream()
                
                await asyncio.gather(send_audio_events(), handler.handle_events())

                final_transcript = handler.get_transcript()
                if final_transcript:
                    self.logger.info(LogComponent.TRANSCRIBE, f"Final transcript: {final_transcript}")
                    self.transcribe_to_bedrock.put(final_transcript)             
                    self.metrics.end_metric(MetricType.API_LATENCY, "transcribe_stream")

            except (BotoCoreError, ClientError) as e:
                self.logger.error(LogComponent.TRANSCRIBE, f"AWS Transcribe error: {e}")
            except Exception as e:
                self.logger.error(LogComponent.TRANSCRIBE, f"Unexpected error: {e}")
        
    def stop(self):
        super().stop()