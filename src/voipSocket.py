import asyncio
import json
import queue
from quart import Quart, websocket, request, Response, copy_current_websocket_context
# from main import ConversationManager
from config import AppConfig, api_request_list
from Orchestrate import ConversationController

app = Quart(__name__)

class WebsocketStreams:
    def __init__(self):
        self.input_stream = queue.Queue()
        self.output_stream = queue.Queue()
    
    async def enqueue_input_stream(self, audio_bytes):
        try:
            self.input_stream.put_nowait(audio_bytes)
        except queue.Full:
            print("Input stream is full")
    
    async def dequeue_input_stream(self):
        # Getting around using asyncio.Queue since we do threading
        while True: 
            try:
                data = self.input_stream.get_nowait()
                return data
            except queue.Empty:
                await asyncio.sleep(0.5)

    async def enqueue_output_stream(self, audio_bytes):
        try:
            self.output_stream.put_nowait(audio_bytes)
        except queue.Full:
            print("Output stream is full")
    
    async def dequeue_output_stream(self):
        # Getting around using asyncio.Queue since we do threading
        while True: 
            try:
                data = self.output_stream.get_nowait()
                return data
            except queue.Empty:
                await asyncio.sleep(0.3)
    
    
@app.websocket("/socket")
async def handle_websocket():
    await websocket.accept()
    websocket_streams = WebsocketStreams()
    
    # Create conversation controller
    conversation = ConversationController(websocket_streams)
    
    async def read_from_websocket():
        while True:
            data = await websocket.receive()
            await websocket_streams.enqueue_input_stream(data)
    
    async def write_to_websocket():
        while True:
            data = await websocket_streams.dequeue_input_stream()
            await websocket.send(data)
    
    read_audio_task = asyncio.create_task(read_from_websocket())
    write_audio_task = asyncio.create_task(write_to_websocket())
    
    try:
        # Run conversation controller in executor and gather with websocket tasks
        loop = asyncio.get_event_loop()
        await asyncio.gather(
            read_audio_task,
            write_audio_task,
            loop.run_in_executor(None, conversation.start_conversation)
        )
    finally:
        # conversation.is_running = False
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