import asyncio
import threading
import pyaudio
import numpy as np
import boto3
import keyboard
import time
import json
from queue import Queue
from botocore.config import Config
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from config import AppConfig

class AudioRecorder:
  def __init__(self, chunk_size=2048*2, sample_rate=16000):
    self.chunk_size = chunk_size
    self.sample_rate = sample_rate
    self.audio = pyaudio.PyAudio()
    self.stream = None
    self.is_recording = False
        
  def start_recording(self):
    print("IN AUDIO RECORDER, STARTING STREAM")
    self.stream = self.audio.open(
      format=pyaudio.paInt16,  # Use 16-bit PCM
      channels=1,
      rate=self.sample_rate,
      input=True,
      frames_per_buffer=self.chunk_size
    )
    self.is_recording = True
      
  def stop_recording(self):
    if self.stream:
      self.stream.stop_stream()
      self.stream.close()
    self.is_recording = False
        
  def read_audio_chunk(self):
    print("IN AUDIO RECORDER, READING CHUNK")
    if self.is_recording:
      data = self.stream.read(self.chunk_size)
      return np.frombuffer(data, dtype=np.int16)  # Read as 16-bit PCM
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
    print(f"Current dB level: {db}")
    print(f"Chunk length: {len(audio_chunk)}")
    if db < self.threshold:
      self.silence_counter += len(audio_chunk)
    else:
      self.silence_counter = 0
    
    print(f"CHECKING SILENCE {self.silence_counter} >= {self.min_silence_samples}")
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
    print(f"transcript handler invoked with results arr of len {len(results)}")
    for result in results:
      print(f"returned result in transcript handler: {result}")
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
  def __init__(self):
    config = Config(
        read_timeout=30,
        retries={'max_attempts': 2}
    )
    #self.client = boto3.client('bedrock-runtime', config=config)
        
  async def get_response(self, text):
    # prompt = {
    #     "prompt": text,
    #     "max_tokens_to_sample": 500,
    #     "temperature": 0.7,
    #     "top_p": 0.9,
    # }
      
    # response = self.client.invoke_model(
    #   modelId="anthropic.claude-v2",
    #   body=json.dumps(prompt)
    # )
    # response_body = json.loads(response['body'].read())
    # return response_body['completion']
    return "no response"
  
class ConversationManager:
  def __init__(self):
    self.recorder = AudioRecorder()
    self.silence_detector = SilenceDetector()
    self.transcribe_client = TranscribeStreamingClient(region="us-west-2")
    self.bedrock = BedrockInference()
    self.interrupt_event = threading.Event()
  
  # Detect enter being pressed in a seperate thread
  def setup_interrupt_detection(self):
    def on_enter(event):
      if event.name == 'enter':
        self.interrupt_event.set()
            
    keyboard.on_press(on_enter)
    
  async def process_audio_stream(self):
    print("starting transcription stream")
    stream = await self.transcribe_client.start_stream_transcription(
      language_code="en-US",
      media_sample_rate_hz=16000,
      media_encoding="pcm"
    )
    
    handler = TranscriptionHandler(stream.output_stream)

    async def write_chunks():
      print("processing audio chunks") 
      while self.recorder.is_recording and not self.interrupt_event.is_set():
        chunk = self.recorder.read_audio_chunk()
        if chunk is not None:
          print("sending audio chunk to transcript and giving up lock with await")
          await stream.input_stream.send_audio_event(audio_chunk=chunk.tobytes())
          print("Got response from transcribe, sending to silence check")
          if self.silence_detector.is_silent(chunk):
            print("detected silence")
            break
        # Yield control so we can asynchronously process transcription events as well. 
        # while loop eats up cpu time
        print("sleeping now for 0.1 sec")
        await asyncio.sleep(0.1)
          
      await stream.input_stream.end_stream()

    loop = asyncio.get_event_loop()
    await asyncio.gather(write_chunks(), handler.handle_events())
    print("returning out of processing audio stream and returning transcript")
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
        continue
            
      print("\nWaiting for next input...")
      
  
async def main():
  config = AppConfig(
    aws_region="us-west-2",
    model_id="meta.llama3-1-70b-instruct-v1:0",
    language_code="en-US"
  )
  
  manager = ConversationManager()
  await manager.run_conversation()
  


if __name__ == "__main__":
  asyncio.run(main())
