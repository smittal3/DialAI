import asyncio
import json
import time
from quart import Quart, websocket, request, Response, copy_current_websocket_context
# from main import ConversationManager
from config import AppConfig, api_request_list

app = Quart(__name__)

class WebsocketStreams:
  def __init__(self):
    self.input_stream = asyncio.Queue()
    self.output_stream = asyncio.Queue()
    
  async def enqueue_input_stream(self, audio_bytes):
    await self.input_stream.put(audio_bytes)
    
  async def dequeue_input_stream(self):
    data = await self.input_stream.get()
    return data if data is not None else b''  # Ensure we never return None
    
  async def enqueue_output_stream(self, audio_bytes):
    await self.output_stream.put(audio_bytes)
  
  async def dequeue_output_stream(self):
    data = await self.output_stream.get()
    return data if data is not None else b''  # Ensure we never return None

    
# currently global var, should make per-instance once we do concurrent calling

@app.websocket("/socket")
async def handle_websocket():
  await websocket.accept()
  websocket_streams = WebsocketStreams()
  
  async def read_from_websocket():
    while True:
      data = await websocket.receive()
      await websocket_streams.enqueue_input_stream(data)
        
  async def write_to_websocket():
    while True:
      data = await websocket_streams.dequeue_input_stream()
      await websocket.send(data)
  
  # Use the current event loop explicitly
  loop = asyncio.get_event_loop()

  # Create tasks on the same event loop
  read_audio_task = loop.create_task(read_from_websocket())
  write_audio_task = loop.create_task(write_to_websocket())
  # read_audio_task = asyncio.create_task(read_from_websocket())
  # write_audio_task = asyncio.create_task(write_to_websocket())
  
  try: 
    await asyncio.gather(read_audio_task, write_audio_task)
  finally: 
    read_audio_task.cancel()
    write_audio_task.cancel()
    
    
@app.route("/webhooks/events", methods=["POST", "GET"])
async def events():
  return "Event received", 200

@app.route("/webhooks/answer")
async def answer_call():
  ncco = [
    {
      "action": "talk",
      "text": "Welcome, now connecting you to our agent Rachel.",
    },
    {
      "action": "connect",
      "from": "Vonage",
      "endpoint": [
        {
          "type": "websocket",
          "uri": f"wss://{request.host}/socket",
          "content-type": "audio/l16;rate=16000",
        }
      ],
    },
  ]

  return Response(json.dumps(ncco), content_type='application/json')