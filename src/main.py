import asyncio
import threading
import pyaudio
import numpy as np
import boto3
import keyboard
import time
import json
from collections import defaultdict
from queue import Queue, Empty
from botocore.config import Config
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from config import AppConfig, api_request_list

class SpeechGenerator():
  def __init__(self, config, speech_complete, bedrock_complete, interrupt_event):
    self.config = config
    self.polly = boto3.client('polly', region_name=config.aws_region)
    self.audio = pyaudio.PyAudio()
    self.stream = None
    self.chunk = 1024
    self.is_processing = False
    self.text_queue = Queue()
    self.generating_thread = None
    self.complete = speech_complete
    self.bedrock_complete = bedrock_complete
    self.interrupt = interrupt_event
    
  def start_generating(self):
    self.is_processing = True
    self.stream = self.audio.open(
      format=pyaudio.paInt16,
      channels=1,
      rate=16000,
      output=True
    )
    self.generating_thread = threading.Thread(target=self._generate_audio)
    self.generating_thread.start()
    
  def _generate_audio(self):
    while self.is_processing:
      try:
        # add a larger timeout to give bedrock time to stream its response
        # Still a little janky. At 0.5 we are skipping, at 1 theres too much delay. Need 
        # some sort of diff mechanism.
        text = self.text_queue.get(timeout=0.1)
        if text:
          # print(f"getting from queue of size {self.text_queue.qsize()}")
          response = self.polly.synthesize_speech(
            Text=text,
            Engine=self.config.polly['engine'],
            LanguageCode=self.config.polly['language'],
            VoiceId=self.config.polly['voice'],
            OutputFormat=self.config.polly['outputFormat']
          )
          
          stream = response['AudioStream']
          while True:
            audio_chunk = stream.read(self.chunk)
            if not audio_chunk:
              break
            # signal completion prematurely
            if self.interrupt.is_set():
              self.complete.set()
            self.stream.write(audio_chunk)
      except Empty:
        # If queue is empty, set the completion flag
        if self.bedrock_complete.is_set():
          self.complete.set()
        continue
      except Exception as e:
        print(f"Error in Polly processing: {e}")

    
  def stop_generating(self):
    self.is_processing = False
    if self.generating_thread:
      self.generating_thread.join()
    if self.stream:
      self.stream.stop_stream()
      self.stream.close()
  
  def add_text_to_queue(self, text):
    self.text_queue.put(text)
  
  def clear_queue(self):
    with self.text_queue.mutex:
      self.text_queue.queue.clear()
        

class AudioRecorder:
  def __init__(self, chunk_size=2048*2, sample_rate=16000):
    self.chunk_size = chunk_size
    self.sample_rate = sample_rate
    self.audio = pyaudio.PyAudio()
    self.stream = None
    self.is_recording = False
    self.audio_queue = Queue()
    self.recording_thread = None

  def start_recording(self):
    #print("IN AUDIO RECORDER, STARTING STREAM")
    self.stream = self.audio.open(
      format=pyaudio.paInt16,  # Use 16-bit PCM
      channels=1,
      rate=self.sample_rate,
      input=True,
      frames_per_buffer=self.chunk_size
    )
    self.is_recording = True
    self.recording_thread = threading.Thread(target=self._record_audio)
    self.recording_thread.start()
  
  def _record_audio(self):
    while self.is_recording:
      try:
        data = self.stream.read(self.chunk_size, exception_on_overflow=False)
        self.audio_queue.put(np.frombuffer(data, dtype=np.int16))
      except Exception as e:
        print(f"Audio recording error: {e}")
        break
    
  def stop_recording(self):
    self.is_recording = False
    if self.recording_thread:
      self.recording_thread.join()
    if self.stream:
      self.stream.stop_stream()
      self.stream.close()
  
  def clear_audio_queue(self):
    # get lock on queue and clear it
    with self.audio_queue.mutex:
      self.audio_queue.queue.clear()
        
  def get_audio_chunk(self, timeout=0.1):
    try:
      return self.audio_queue.get(timeout=timeout)
    except Empty:
      return None
  

class SilenceDetector:
  def __init__(self, long_silence_indicator, threshold=-30, min_silence_duration=1.0, sample_rate=16000):
    self.threshold = threshold
    self.min_silence_samples = int(min_silence_duration * sample_rate)
    self.silence_counter = 0
    self.long_silence_indicator = long_silence_indicator
        
  def is_silent(self, audio_chunk):
    if audio_chunk is None:
      return False

    # normalize to float32 values
    audio_chunk_normalized = audio_chunk / 32768.0

    db = 20 * np.log10(np.max(np.abs(audio_chunk_normalized)) + 1e-10)
    #print(f"Current dB level: {db}")
    #print(f"Chunk length: {len(audio_chunk)}")
    if db < self.threshold:
      self.silence_counter += len(audio_chunk)
    else:
      self.silence_counter = 0
      self.long_silence_indicator.clear()
      
    #print(f"CHECKING SILENCE {self.silence_counter} >= {self.min_silence_samples}")
    return self.silence_counter >= self.min_silence_samples
    
  def reset(self):
    self.silence_counter = 0
    self.long_silence_indicator.clear()
    
    
class TranscriptionHandler(TranscriptResultStreamHandler):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.transcript_buffer = []
      
  # Everytime a transcript event is returned from AWS, append to buffer
  async def handle_transcript_event(self, transcript_event: TranscriptEvent):
    results = transcript_event.transcript.results
    #print(f"transcript handler invoked with results arr of len {len(results)}")
    for result in results:
      #print(f"returned result in transcript handler: {result}")
      if not result.is_partial:
        transcript = result.alternatives[0].transcript
        self.transcript_buffer.append(transcript)
        print(f"transcript updated: {self.transcript_buffer}")

  
  # return current transcript and clear the buffer
  def get_transcript(self):
    transcript = ' '.join(self.transcript_buffer)
    self.transcript_buffer = []
    return transcript

class BedrockInference:
  def __init__(self, config, interrupt, bedrock_context, bedrock_complete):
    bedrock_config = Config(
      read_timeout=30,
      retries={'max_attempts': 2}
    )
    self.client = boto3.client('bedrock-runtime', config=bedrock_config, region_name=config.aws_region)
    self.config = config
    self.interrupt_event = interrupt
    self.context = bedrock_context
    self.bedrock_complete = bedrock_complete
        
  async def get_response(self, text):
    self.context.add_user_input(text)
    body = self.define_body(text)
    
    try: 
      body_json = json.dumps(body)
      model_params = api_request_list[self.config.model_id]
      response = self.client.invoke_model_with_response_stream(
        body=body_json, 
        modelId=model_params['modelId'], 
        accept=model_params['accept'], 
        contentType=model_params['contentType']
      )
      full_response = ""
      bedrock_stream = response.get('body')
      async for chunk in self.process_stream(bedrock_stream):
        full_response += chunk
        yield chunk
        if self.interrupt_event.is_set():
          print("\nInference interrupted!")
          break
      
      self.context.add_bedrock_output(full_response)
      self.bedrock_complete.set()
                
    except Exception as e: 
      print(f"Bedrock Error {e}")
      yield "no response, error"
  
  def define_body(self, text):
    body = api_request_list[self.config.model_id]['body']
    body['prompt'] = self.context.get_context()
    print(f"prompt is {body}")
    return body
  
  async def process_stream(self, bedrock_stream):
    buffer = ''
    
    if bedrock_stream:
      for event in bedrock_stream:
        chunk = event.get('chunk')
        if chunk:
          chunk_obj = json.loads(chunk.get('bytes').decode())
          text = chunk_obj['generation']
          buffer += text
          
          # Stream out complete sentences
          while '.' in buffer:
            sentence, buffer = buffer.split('.', 1)
            yield sentence + '.'
                  
      # Yield any remaining text in buffer
      if buffer:
        yield buffer + ('.' if not buffer.endswith('.') else '')
    
class BedrockContext: 
  def __init__(self, config):
    self.history = []
    self.formatted_context= f"<|begin_of_text|><|start_header_id|>system \
                              <|end_header_id|>\n\n{config.system_prompt}<|eot_id|>\n"
  
  def add_user_input(self, user_input):
    self.history.append({"role":"user", "message": user_input})
    
    user_turn = f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>"
    partial_model_tag = f"<|start_header_id|>assistant<|end_header_id|>"
    self.formatted_context += f"{user_turn}\n{partial_model_tag}\n"
    
  def add_bedrock_output(self, bedrock_output):
    self.history.append({"role":"assistant", "message": bedrock_output})
    self.formatted_context += f"\n{bedrock_output}<|eot_id|>\n"
  
  def get_context(self):
    return self.formatted_context


class ConversationManager:
  def __init__(self, config):
    self.interrupt_event = threading.Event()
    self.speech_complete = threading.Event()
    self.bedrock_complete = threading.Event()
    self.long_silence_indicator = threading.Event()
    self.recorder = AudioRecorder()
    self.speech_generator = SpeechGenerator(config, self.speech_complete, self.bedrock_complete, self.interrupt_event)
    self.silence_detector = SilenceDetector(self.long_silence_indicator)
    self.transcribe_client = TranscribeStreamingClient(region=config.aws_region)
    self.bedrock_context = BedrockContext(config)
    self.bedrock = BedrockInference(config, self.interrupt_event, self.bedrock_context, self.bedrock_complete)
    # Flag to control silence detection during bedrock inference
  
  # Detect enter being pressed in a seperate thread
  def setup_interrupt_detection(self):
    def on_enter(event):
      if event.name == 'enter':
        self.interrupt_event.set()
            
    keyboard.on_press(on_enter)
    
  async def process_audio_stream(self):
    #print("starting transcription stream")
    stream = await self.transcribe_client.start_stream_transcription(
      language_code="en-US",
      media_sample_rate_hz=16000,
      media_encoding="pcm"
    )
    
    handler = TranscriptionHandler(stream.output_stream)

    async def write_chunks():
      #print("processing audio chunks") 
      while self.recorder.is_recording and not self.interrupt_event.is_set():
        chunk = self.recorder.get_audio_chunk()
        # If not silence and valid chunk, send audio events for transcribe
        if chunk is not None:
          #print("sending audio chunk to transcript and giving up lock with await")
          if self.silence_detector.is_silent(chunk):
            # if a conversational turn is over and continuous silence is detected, block. Otherwise if "new" silence is detected, break
            if self.long_silence_indicator.is_set():
              continue
            print("detected silence")
            break
          
          await stream.input_stream.send_audio_event(audio_chunk=chunk.tobytes())
          #print("Got response from transcribe, sending to silence check")
          
        # Yield control so we can asynchronously process transcription events as well. 
        # while loop eats up cpu time
        await asyncio.sleep(0.1)
          
      await stream.input_stream.end_stream()

    loop = asyncio.get_event_loop()
    await asyncio.gather(write_chunks(), handler.handle_events())
    #print("returning out of processing audio stream and returning transcript")
    return handler.get_transcript()
  
  async def run_conversation(self):
    self.setup_interrupt_detection()
        
    while True:
      print("\nListening... (Press Enter to interrupt)")
      self.recorder.start_recording()
      self.interrupt_event.clear()
      self.speech_complete.clear()
      self.bedrock_complete.clear()
        
      try:
        # Get all transcribed data so far and send to inference for bedrock
        print("IN MAIN, process audio stream operation until silence")
        start_recording = time.time()
        transcript = await self.process_audio_stream()
        self.recorder.stop_recording()
        end_recording = time.time()
        if transcript:
          self.speech_generator.start_generating()
          print(f"\nTranscript: {transcript} \n Bedrock Response:")
          async for response_chunk in self.bedrock.get_response(transcript):
            self.speech_generator.add_text_to_queue(response_chunk)
            print(response_chunk, end='', flush=True)

          # Wait until speech synthesis is complete, interrupt event is handled in that thread
          self.speech_complete.wait()
          self.speech_generator.stop_generating()
          
          response_bedrock = time.time()
          recording_time = end_recording - start_recording
          bedrock_time = response_bedrock - end_recording
          total_time = response_bedrock - start_recording 
          
          print(f"\nRecording time: {recording_time:.6f} seconds")
          print(f"\nBedrock time: {bedrock_time:.6f} seconds")
          print(f"\nTotal time: {total_time:.6f} seconds")
        
      except Exception as e:
        print(f"Error: {e}")
      finally:
        self.recorder.stop_recording()
        self.speech_generator.stop_generating()
        # Set this to block until a user is ready to speak again (detected via noise)
        self.long_silence_indicator.set()
        # These queues are clearing here under the assumption that the outer while loop is only iterating once per conversational turn
        # and our asyncio.gather() on speech generation is valid. Currently we are using a janky timeout mechanism. This should be 
        # reevaluated. Without this clear here, the mic is picking up audio data from polly's output. This is not intended. Ideally
        # we find a way to suspend the audio recording thr
        self.recorder.clear_audio_queue()
        self.speech_generator.clear_queue()   
      
      if self.interrupt_event.is_set():
        print("\nInterrupted! Starting new conversation...") 
        continue
               
      print("\nWaiting for next input...")
      
  
async def main():
  config = AppConfig(
    aws_region="us-west-2",
    model_id="meta.llama3-1-70b-instruct-v1:0",
    language_code="en-US"
  )
  
  manager = ConversationManager(config)
  await manager.run_conversation()
  


if __name__ == "__main__":
  asyncio.run(main())
