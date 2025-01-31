import asyncio
import json
import queue
import threading
import signal
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
                await asyncio.sleep(0.2)

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
                await asyncio.sleep(0.2)
    
    
@app.websocket("/socket")
async def handle_websocket():
    # first receive is not audio data
    await websocket.receive()
    
    # setup streams and synchronization events
    websocket_streams = WebsocketStreams()
    system_interrupt = threading.Event()
    user_interrupt = threading.Event()

    config = AppConfig(
        aws_region="us-west-2",
        model_id="meta.llama3-1-70b-instruct-v1:0",
        language_code="en-US"
    )

    # Create conversation controller
    controller = ConversationController(websocket_streams, system_interrupt, user_interrupt, config)
    
    async def read_from_websocket():
        while True:
            data = await websocket.receive()
            await websocket_streams.enqueue_input_stream(data)
            if system_interrupt.is_set():
                print("Exiting websocket read")
                break
    
    async def write_to_websocket():
        buffer = bytearray()
        while True:
            data = await websocket_streams.dequeue_output_stream()  
            buffer.extend(data)
            # At 16000 Hz, ensure we are sending 640 bytes per packet
            while len(buffer) > 640:
                data = buffer[:640]
                buffer = buffer[640:]
                await websocket.send(data)

            if system_interrupt.is_set():
                print("Exiting websocket write")
                break
            
            if user_interrupt.is_set():
                user_interrupt.clear()
                out = websocket_streams.output_stream
                with out.mutex:
                    out.queue.clear()

            

    def signal_handler():
        system_interrupt.set()
        controller.stop_conversation() 
    
    read_audio_task = asyncio.create_task(read_from_websocket())
    write_audio_task = asyncio.create_task(write_to_websocket())
    
    try:
        # Run conversation controller in executor and gather with websocket tasks
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, signal_handler)

        await asyncio.gather(
            read_audio_task,
            write_audio_task,
            loop.run_in_executor(None, controller.start_conversation)
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
            "text": "Welcome.",
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