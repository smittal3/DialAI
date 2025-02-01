import threading
import queue
import time
from VoiceDetection import VoiceDetection
from Transcribe import Transcribe
from Inference import LLMInference, BedrockContext
from SpeechGenerator import SpeechGenerator

class ConversationController:
    def __init__(self, websocket_streams, system_interrupt, user_interrupt, config):
        self.user_interrupt = user_interrupt
        self.silence_indicator = threading.Event()
        self.system_interrupt = system_interrupt
        self.bedrock_complete = threading.Event()
        self.speech_complete = threading.Event()
        self.is_running = True
        self.config = config

        # shared state
        self.vad_to_transcribe = queue.Queue()
        self.transcribe_to_bedrock = queue.Queue()
        self.bedrock_to_stt = queue.Queue()

        self.context = BedrockContext(config)
        

        # Initialize components
        self.vad = VoiceDetection(websocket_streams.input_stream, websocket_streams.output_stream,
                                  self.vad_to_transcribe, self.silence_indicator, self.user_interrupt)

        self.transcribe = Transcribe(websocket_streams.input_stream, self.vad_to_transcribe, self.transcribe_to_bedrock, 
                                    self.user_interrupt, self.silence_indicator, self.system_interrupt, self.config)
        
        self.inference = LLMInference(self.config, self.user_interrupt, self.context, self.bedrock_complete, 
                                      self.transcribe_to_bedrock, self.bedrock_to_stt)

        self.speech_generator = SpeechGenerator(self.config, self.user_interrupt, self.bedrock_complete, 
                                                self.speech_complete, self.bedrock_to_stt, websocket_streams.output_stream)

    def start_conversation(self):
        print("Starting conversation in controller")
        try:
            # These two block in silence, one thread over the application
            self.vad.start()
            self.transcribe.start() 
            while self.is_running:
                # Wait for speech, then for silence
                self.user_interrupt.wait()
                self.silence_indicator.wait()
                self.inference.start()
                self.speech_generator.start()

                print("Waiting for bedrock complete")
                self.bedrock_complete.wait()
                self.inference.stop()

                print("Waiting for speech complete")
                self.speech_complete.wait()
                self.speech_generator.stop()

                self.speech_complete.clear()
                self.bedrock_complete.clear()
        except Exception as e:
            self.stop_conversation()

    def stop_conversation(self):
        print("Stopping conversation in controller")
        self.is_running = False
        self.user_interrupt.set()
        self.vad.stop()
        self.transcribe.stop()
        self.inference.stop()
        self.speech_generator.stop()    