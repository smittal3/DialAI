import torch
import numpy as np
import queue
import threading
from typing import Optional
from Logger import Logger, LogComponent
from Metrics import Metrics, MetricType

class VoiceDetection:
    def __init__(self, 
                 input_queue: queue.Queue,
                 output_queue: queue.Queue,
                 vad_to_transcribe: queue.Queue,
                 silence_indicator: threading.Event,
                 user_interrupt: threading.Event,
                 sample_rate: int = 16000,
                 threshold: int = 0.6):
    
        self.input_queue = input_queue
        self.vad_to_transcribe = vad_to_transcribe
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_indicator = silence_indicator
        self.user_interrupt = user_interrupt
        self.logger = Logger()
        self.metrics = Metrics()
        
        # Initialize Silero VAD
        self.logger.info(LogComponent.VAD, "Initializing Silero VAD model")
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                     model='silero_vad',
                                     force_reload=False)
        self.model.eval()
        
        # State management
        self.should_stop = False
        self.thread: Optional[threading.Thread] = None
        self.is_speaking = False
        
        # Window for silero is 512 samples at 16000 Hz
        self.window_size = 512
        self.silence_counter = 0
        self.min_silence_samples = 35
        self.speech_samples = 0
        self.min_speech_samples = 10
        self.logger.info(LogComponent.VAD, "Voice Detection initialized successfully")
    
    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        
        self.logger.info(LogComponent.VAD, "Starting voice detection thread")
        self.should_stop = False
        self.thread = threading.Thread(target=self._process_stream)
        self.thread.start()

    def stop(self):
        self.logger.info(LogComponent.VAD, "Stopping voice detection thread")
        self.should_stop = True
        if self.thread is not None:
            self.thread.join()
            self.thread = None
    
    def _detect_speech(self, audio_chunk: np.ndarray) -> bool:
        tensor = torch.from_numpy(audio_chunk).float()
        speech_prob = self.model(tensor, self.sample_rate).item()
        return speech_prob > self.threshold
    
    def _process_stream(self):
            self.logger.info(LogComponent.VAD, "Starting audio processing stream")
            audio_buffer = np.array([], dtype=np.int16)
            audio_buffer_normalized = np.array([], dtype=np.int16)
            byte_buffer = bytearray()
            first_chunk = True

            while not self.should_stop:
                try:
                    # Chunks are currently 640 bytes each
                    chunk = self.input_queue.get(timeout=0.01)
                    byte_buffer.extend(chunk)
                    # chunk should now be 1/2 of previous size, each byte pair is one sample
                    chunk = np.frombuffer(chunk, dtype=np.int16)
                    audio_buffer = np.concatenate([audio_buffer, chunk])

                    # normalize to -1.0 to 1.0
                    chunk = chunk.astype(np.float32) / 32768.0
                    audio_buffer_normalized = np.concatenate([audio_buffer_normalized, chunk])
                    
                    # Process complete windows (512 samples at 16000 Hz)
                    while len(audio_buffer_normalized) >= self.window_size:
                        window_normalized = audio_buffer_normalized[:self.window_size]
                        audio_buffer_normalized = audio_buffer_normalized[self.window_size:]
                        
                        window = audio_buffer[:self.window_size]
                        audio_buffer = audio_buffer[self.window_size:]

                        is_speech = self._detect_speech(window_normalized)

                        if is_speech:
                            self.silence_counter = 0
                            self.speech_samples += 1
                            self.is_speaking = True
                        else:
                            self.silence_counter += 1
                       
                        if self.is_speaking:
                            if first_chunk:
                                self.metrics.start_metric(MetricType.TRANSCRIPTION, "first_chunk_vad_to_transcribe")
                                first_chunk = False
                            self.vad_to_transcribe.put(window)
                    
                    # This is still 5 frames after the user has supposedly stopped speaking
                    if self.silence_counter == 10:
                        self.metrics.start_metric(MetricType.THREAD_LIFETIME, "*****response_time_from_silence*****")

                    if self.speech_samples >= self.min_speech_samples:
                        self.user_interrupt.set()
                        self.silence_indicator.clear()

                    if self.is_speaking and self.silence_counter >= self.min_silence_samples:
                        self.silence_indicator.set()
                        self.user_interrupt.clear()
                        self.speech_samples = 0
                        self.is_speaking = False
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(LogComponent.VAD, f"Error in processing: {e}")
                    continue
