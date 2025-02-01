import torch
import numpy as np
import queue
import threading
from typing import Optional

class VoiceDetection:
    def __init__(self, 
                 input_queue: queue.Queue,
                 output_queue: queue.Queue,
                 vad_to_transcribe: queue.Queue,
                 silence_indicator: threading.Event,
                 user_interrupt: threading.Event,
                 sample_rate: int = 16000,
                 threshold: int = 0.4):
    
        self.input_queue = input_queue
        #self.output_queue = output_queue
        self.vad_to_transcribe = vad_to_transcribe
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_indicator = silence_indicator
        self.user_interrupt = user_interrupt
        
        # Initialize Silero VAD
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
        self.min_silence_samples = 50
        self.speech_samples = 0
        self.min_speech_samples = 10
    
    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        
        # print("[VAD] Starting voice detection thread")
        self.should_stop = False
        self.thread = threading.Thread(target=self._process_stream)
        self.thread.start()

    def stop(self):
        # print("[VAD] Stopping voice detection thread")
        self.should_stop = True
        if self.thread is not None:
            self.thread.join()
            self.thread = None
    
    def _detect_speech(self, audio_chunk: np.ndarray) -> bool:
        tensor = torch.from_numpy(audio_chunk).float()
        speech_prob = self.model(tensor, self.sample_rate).item()
        # print(f"[VAD] Speech probability: {speech_prob:.3f}")
        return speech_prob > self.threshold
    
    def _process_stream(self):
        # print("[VAD] Starting audio processing stream")
        audio_buffer = np.array([], dtype=np.int16)
        audio_buffer_normalized = np.array([], dtype=np.int16)
        byte_buffer = bytearray()

        while not self.should_stop:
            try:
                # Chunks are currently 640 bytes each
                chunk = self.input_queue.get(timeout=1)
                # print(f"[VAD] Received audio chunk of size: {len(chunk)}")
                byte_buffer.extend(chunk)
                # chunk should now be 1/2 of previous size, each byte pair is one sample
                chunk = np.frombuffer(chunk, dtype=np.int16)
                audio_buffer = np.concatenate([audio_buffer, chunk])

                # normalize to -1.0 to 1.0
                chunk = chunk.astype(np.float32) / 32768.0
                audio_buffer_normalized = np.concatenate([audio_buffer_normalized, chunk])
                
                # Process complete windows (512 samples at 16000 Hz)
                while len(audio_buffer_normalized) >= self.window_size:
                    # Get window and update buffer
                    window_normalized = audio_buffer_normalized[:self.window_size]
                    audio_buffer_normalized = audio_buffer_normalized[self.window_size:]
                    
                    window = audio_buffer[:self.window_size]
                    audio_buffer = audio_buffer[self.window_size:]

                    window_bytes = byte_buffer[:self.window_size * 2]
                    byte_buffer = byte_buffer[self.window_size * 2:]

                    is_speech = self._detect_speech(window_normalized)

                    if is_speech:
                        self.silence_counter = 0
                        self.speech_samples += 1
                        self.is_speaking = True
                    else:
                        self.silence_counter += 1
                   
                    if self.is_speaking:
                        # currently outputting window_bytes to create echo server
                        # otherwise, should be outputting audio_buffer for STT
                        # self.output_queue.put(window_bytes)
                        # This is all we need to do
                        # TO:DO: Remove adding window_bytes to output_queue
                        self.vad_to_transcribe.put(window)

                # only do once
                if self.speech_samples >= self.min_speech_samples:
                    self.user_interrupt.set()
                    self.silence_indicator.clear()

                # After processing the window, check if we should trigger a silence event
                if self.is_speaking and self.silence_counter >= self.min_silence_samples:
                    print("[VAD] Silence detected")
                    self.silence_indicator.set()
                    self.user_interrupt.clear()
                    self.speech_samples = 0
                    self.is_speaking = False

             
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VAD] Error in processing: {e}")
                continue
