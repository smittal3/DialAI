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
from Logger import LogComponent, Logger
class ConversationController:
    def __init__(self, websocket_streams, system_interrupt, user_interrupt, config):
        self.user_interrupt = user_interrupt
        self.silence_indicator = threading.Event()
        self.system_interrupt = system_interrupt
        self.bedrock_complete = threading.Event()
        self.speech_complete = threading.Event()
        self.transcribe_complete = threading.Event()
        self.is_running = True
        self.config = config
        self.metrics = Metrics()
        self.logger = Logger()
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

        self.transcribe = Transcribe(self.vad_to_transcribe, self.transcribe_to_bedrock, self.transcribe_complete, self.user_interrupt, 
                                     self.silence_indicator, self.system_interrupt, self.config)
        
        self.inference = LLMInference(self.config, self.user_interrupt, self.context, self.bedrock_complete, 
                                      self.transcribe_complete, self.transcribe_to_bedrock, self.bedrock_to_stt)

        self.speech_generator = SpeechGenerator(self.config, self.user_interrupt, self.bedrock_complete, 
                                                self.speech_complete, self.bedrock_to_stt, websocket_streams.output_stream)

    def start_conversation(self):
        print("Starting conversation in controller")
        self.start_time = datetime.now()
        try:
            # These two block in silence, one thread over the application
            self.transcribe.start() 
            self.vad.start()
            
            # do one inference call to warm up 
            self.transcribe_to_bedrock.put("I am going to connect you to the call, are you ready? Respond with yes or no.")
            self.transcribe_complete.set()
            self.inference.start()
            self.bedrock_complete.wait()
            self.inference.stop()
            self.transcribe_complete.clear()
            with self.bedrock_to_stt.mutex:
                self.bedrock_to_stt.queue.clear()
            self.bedrock_complete.clear()
            
            while self.is_running:
                # Wait for speech, then for silence
                self.user_interrupt.wait()
                
                # Start inference before waiting for silence to save on setup time of stream
                self.silence_indicator.wait()
                self.logger.debug(LogComponent.SYSTEM, "Starting inference")
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
        finally:
            self.stop_conversation()

    def stop_conversation(self):
        print("Stopping conversation in controller")
        self.is_running = False
        
        # Set all events to unblock waiting threads
        self.user_interrupt.set()
        self.system_interrupt.set()
        self.silence_indicator.set()
        self.bedrock_complete.set()
        self.transcribe_complete.set()
        self.speech_complete.set()  

        # Clear all queues to unblock any waiting gets/puts
        for q in [self.vad_to_transcribe, self.transcribe_to_bedrock, 
                  self.bedrock_to_stt]:
            with q.mutex:
                q.queue.clear()

        # Stop all threads with a timeout
        threads = [self.vad, self.transcribe, self.inference, self.speech_generator]
        for thread in threads:
            thread.stop()
            if thread.thread and thread.thread.is_alive():
                thread.thread.join(timeout=1)  # Give each thread 2 seconds to stop
        
        # Store conversation data
        self.end_time = datetime.now()
        # try:
        #     self.database.store_conversation(
        #         self.start_time,
        #         self.end_time,
        #         self.context.history,
        #     )
        # except Exception as e:
        #     print(f"Error storing conversation: {e}")
        # finally:
        #     self.database.close()