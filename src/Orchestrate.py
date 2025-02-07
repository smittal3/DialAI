import threading
import queue
import time
from datetime import datetime
from VoiceDetection import VoiceDetection
from Transcribe import Transcribe
from Inference import LLMInference, BedrockContext
from SpeechGenerator import SpeechGenerator
from Database import Database
from Metrics import Metrics, MetricType

class ConversationController:
    def __init__(self, websocket_streams, system_interrupt, user_interrupt, config):
        self.user_interrupt = user_interrupt
        self.silence_indicator = threading.Event()
        self.system_interrupt = system_interrupt
        self.bedrock_complete = threading.Event()
        self.speech_complete = threading.Event()
        self.is_running = True
        self.config = config
        self.metrics = Metrics()
        self.start_time = None
        self.end_time = None
        
        # Initialize database
        self.database = Database()
        
        # shared state
        self.vad_to_transcribe = queue.Queue()
        self.transcribe_to_bedrock = queue.Queue()
        self.bedrock_to_stt = queue.Queue()

        self.context = BedrockContext(config)
        

        # Initialize components
        self.vad = VoiceDetection(websocket_streams.input_stream, websocket_streams.output_stream,
                                  self.vad_to_transcribe, self.silence_indicator, self.user_interrupt)

        self.transcribe = Transcribe(self.vad_to_transcribe, self.transcribe_to_bedrock, self.user_interrupt, 
                                     self.silence_indicator, self.system_interrupt, self.config)
        
        self.inference = LLMInference(self.config, self.user_interrupt, self.context, self.bedrock_complete, 
                                      self.transcribe_to_bedrock, self.bedrock_to_stt)

        self.speech_generator = SpeechGenerator(self.config, self.user_interrupt, self.bedrock_complete, 
                                                self.speech_complete, self.bedrock_to_stt, websocket_streams.output_stream)

    def start_conversation(self):
        print("Starting conversation in controller")
        self.start_time = datetime.now()
        try:
            # These two block in silence, one thread over the application
            self.transcribe.start() 
            self.vad.start()
            while self.is_running:
                # Wait for speech, then for silence
                self.user_interrupt.wait()
                
                # Start inference before waiting for silence to save on setup time of stream
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
        self.system_interrupt.set()
        self.silence_indicator.set()
        self.bedrock_complete.set()
        self.speech_complete.set()  

        self.vad.stop()
        self.transcribe.stop()
        self.inference.stop()
        self.speech_generator.stop()
        
        # Store conversation data
        self.end_time = datetime.now()
        try:
            self.database.store_conversation(
                self.start_time,
                self.end_time,
                self.context.history,
            )
        except Exception as e:
            print(f"Error storing conversation: {e}")
        finally:
            self.database.close()