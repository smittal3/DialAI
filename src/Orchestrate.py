import threading
import queue
import time
from VoiceDetection import VoiceDetection
from Transcribe import Transcribe

class ConversationController:
    def __init__(self, websocket_streams, system_interrupt, user_interrupt, config):
        self.user_interrupt = user_interrupt
        self.silence_indicator = threading.Event()
        self.system_interrupt = system_interrupt
        self.is_running = True
        self.config = config

        # shared state
        vad_to_transcribe = queue.Queue()
        transcribe_to_bedrock = queue.Queue()

        # Initialize components
        self.vad = VoiceDetection(websocket_streams.input_stream, websocket_streams.output_stream,
                                  vad_to_transcribe, self.silence_indicator, self.user_interrupt)

        self.transcribe = Transcribe(websocket_streams.input_stream, vad_to_transcribe, transcribe_to_bedrock, 
                                    self.user_interrupt, self.silence_indicator, self.system_interrupt, self.config)
        
        

    def start_conversation(self):
        print("Starting conversation in controller")
        self.vad.start()
        self.transcribe.start()
        try:
            while self.is_running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.stop_conversation()

    def stop_conversation(self):
        print("Stopping conversation in controller")
        self.is_running = False
        self.user_interrupt.set()
        self.vad.stop()
        self.transcribe.stop()