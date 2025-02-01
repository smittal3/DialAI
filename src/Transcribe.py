import threading
import queue
import asyncio
from config import AppConfig
from typing import Optional
from botocore.exceptions import BotoCoreError, ClientError
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

class TranscribeHandler(TranscriptResultStreamHandler):
    def __init__(self,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.transcript_buffer = []
        # print("[TranscribeHandler] Initialized")

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # print("[TranscribeHandler] Received transcript event")
        results = transcript_event.transcript.results
        for result in results:
            # print(f"[TranscribeHandler] Processing result: {result}")
            if not result.is_partial:
                transcript = result.alternatives[0].transcript
                self.transcript_buffer.append(transcript)
                print(f"[TranscribeHandler] Added to buffer: {transcript}")


    def get_transcript(self):
        transcript = ' '.join(self.transcript_buffer)
        self.transcript_buffer = []
        return transcript


# Blocks when silent, otherwise triggers transcribe and puts result in output queue to bedrock
class Transcribe:
    def __init__(self, 
                 audio_queue: queue.Queue,
                 vad_to_transcribe: queue.Queue,
                 transcribe_to_bedrock: queue.Queue,
                 user_interrupt: threading.Event,
                 silence_indicator: threading.Event,
                 system_interrupt: threading.Event,
                 config: AppConfig):

        self.audio_queue = audio_queue
        self.vad_to_transcribe = vad_to_transcribe
        self.transcribe_to_bedrock = transcribe_to_bedrock
        self.config = config
        self.user_interrupt = user_interrupt
        self.silence_indicator = silence_indicator
        self.system_interrupt = system_interrupt
        self.thread: Optional[threading.Thread] = None
        self.is_running = True
        # Initialize AWS client
        self.client = TranscribeStreamingClient(region=config.aws_region)

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        def run_async_transcribe():
            asyncio.run(self._transcribe_stream())

        self.thread = threading.Thread(target=run_async_transcribe)
        self.thread.start()

    def stop(self):
        if self.thread is not None:
            print("Stopping transcription thread")
            self.is_running = False
            self.thread.join()
            self.thread = None

    async def _transcribe_stream(self):
        while self.is_running:
            try:
                print("[Transcribe] Waiting for silence indicator to clear")
                while self.silence_indicator.is_set():
                    if self.system_interrupt.is_set():
                        # print("[Transcribe] System interrupt detected while waiting")
                        break
                    await asyncio.sleep(0.2)

                if self.system_interrupt.is_set():
                    # print("[Transcribe] Breaking main loop due to system interrupt")
                    break
                    
                # print("[Transcribe] Starting AWS stream transcription")
                stream = await self.client.start_stream_transcription(
                    language_code=self.config.language_code,
                    media_sample_rate_hz=self.config.sample_rate,
                    media_encoding="pcm"
                )
                
                handler = TranscribeHandler(stream.output_stream)
                stream.handler = handler
                # print("[Transcribe] Handler setup complete")

                async def send_audio_events():
                    # print("[Transcribe] Starting audio event sender")
                    while True:
                        try:
                            data = self.vad_to_transcribe.get(timeout=0.2)
                            # print("[Transcribe] Received audio chunk from VAD")
                            await stream.input_stream.send_audio_event(audio_chunk=data.tobytes())
                        except queue.Empty:
                            if self.silence_indicator.is_set():
                                # print("[Transcribe] Silence detected, ending audio stream")
                                break
                            continue
                        except Exception as e:
                            print(f"[Transcribe] Error in audio stream: {e}")
                            break

                    # print("[Transcribe] Ending input stream")
                    await stream.input_stream.end_stream()
                
                # print("[Transcribe] Starting audio processing")
                await asyncio.gather(send_audio_events(), handler.handle_events())

                final_transcript = handler.get_transcript()
                print(f"[Transcribe] Final transcript produced: {final_transcript}")
                self.transcribe_to_bedrock.put(final_transcript)             

            except (BotoCoreError, ClientError) as e:
                print(f"[Transcribe] AWS Transcribe error: {e}")
            except Exception as e:
                print(f"[Transcribe] Unexpected error: {e}")
            
