import asyncio
import threading
import pyaudio
import numpy as np
import boto3
import keyboard
import time
import json
from queue import Queue, Empty
from botocore.config import Config
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from config import AppConfig, api_request_list

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
    """Get an audio chunk from the queue. Returns None if the queue is empty."""
    try:
      return self.audio_queue.get(timeout=timeout)
    except Empty:
      return None
  

class SilenceDetector:
  def __init__(self, threshold=-30, min_silence_duration=1.0, sample_rate=16000):
    self.threshold = threshold
    self.min_silence_samples = int(min_silence_duration * sample_rate)
    self.silence_counter = 0
        
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
    
    #print(f"CHECKING SILENCE {self.silence_counter} >= {self.min_silence_samples}")
    return self.silence_counter >= self.min_silence_samples
    
  def reset(self):
    self.silence_counter = 0
    
    
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
  def __init__(self, config, interrupt):
    bedrock_config = Config(
      read_timeout=30,
      retries={'max_attempts': 2}
    )
    self.client = boto3.client('bedrock-runtime', config=bedrock_config, region_name=config.aws_region)
    self.config = config
    self.interrupt_event = interrupt
        
  async def get_response(self, text):
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
      
      bedrock_stream = response.get('body')
      bedrock_response = self.extract_response(bedrock_stream)
    except Exception as e: 
      print(f"Bedrock Error {e}")
      bedrock_response = "no response, error"
    
    for audio in bedrock_response:
      if self.interrupt_event.is_set():  # Check for interrupt event
        print("\nInference interrupted!")
        break
      print(audio)

    return bedrock_response
  
  def define_body(self, text):
    body = api_request_list[self.config.model_id]['body']
    body['prompt'] = "Be helpful, keep your responses short under 10 words."
    print(f"prompt is {body}")
    return body
  
  def extract_response(self, bedrock_stream):
    prefix = ''

    if bedrock_stream:
      for event in bedrock_stream:
        chunk = event.get('chunk')
        if chunk:
          chunk_obj = json.loads(chunk.get('bytes').decode())
          text = chunk_obj['generation']

          if '.' in text:
            a = text.split('.')[:-1]
            to_polly = ''.join([prefix, '.'.join(a), '. '])
            prefix = text.split('.')[-1]
            print(to_polly, flush=True, end='')
            yield to_polly
          else:
            prefix = ''.join([prefix, text])

      if prefix != '':
        print(prefix, flush=True, end='')
        yield f'{prefix}.'

      print('\n')
    
  
class ConversationManager:
  def __init__(self, config):
    self.recorder = AudioRecorder()
    self.silence_detector = SilenceDetector()
    self.transcribe_client = TranscribeStreamingClient(region=config.aws_region)
    self.interrupt_event = threading.Event()
    self.bedrock = BedrockInference(config, self.interrupt_event)
  
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
        if chunk is not None:
          #print("sending audio chunk to transcript and giving up lock with await")
          await stream.input_stream.send_audio_event(audio_chunk=chunk.tobytes())
          #print("Got response from transcribe, sending to silence check")
          if self.silence_detector.is_silent(chunk):
            print("detected silence")
            break
        # Yield control so we can asynchronously process transcription events as well. 
        # while loop eats up cpu time
        #print("sleeping now for 0.1 sec")
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
      self.silence_detector.reset()
      self.interrupt_event.clear()
        
      try:
        # Get all transcribed data so far and send to inference for bedrock
        print("IN MAIN, process audio stream operation until silence")
        start_recording = time.time()
        transcript = await self.process_audio_stream()
        end_recording = time.time()
        if transcript:
          print(f"\nTranscript: {transcript}")
          response = await self.bedrock.get_response(transcript)
          response_bedrock = time.time()
          print(f"\nResponse: {response}")
          recording_time = end_recording - start_recording
          bedrock_time = response_bedrock - end_recording
          total_time = response_bedrock - start_recording  # Total time from start to Bedrock response
          
          # Print the durations
          print(f"\nRecording time: {recording_time:.6f} seconds")
          print(f"\nBedrock time: {bedrock_time:.6f} seconds")
          print(f"\nTotal time: {total_time:.6f} seconds")
        
      except Exception as e:
          print(f"Error: {e}")
      finally:
          self.recorder.stop_recording()
      
      if self.interrupt_event.is_set():
        print("\nInterrupted! Starting new conversation...")
        # If conversation is interrupted, clear stale audio from queue
        self.recorder.clear_audio_queue()
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
