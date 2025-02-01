import asyncio
import json
import queue
import threading
import signal
from quart import Quart, websocket, request, Response, copy_current_websocket_context
# from main import ConversationManager
from config import AppConfig, api_request_list
from Orchestrate import ConversationController
from Logger import Logger, LogComponent

app = Quart(__name__)
logger = Logger()

class WebsocketStreams:
    def __init__(self):
        self.input_stream = queue.Queue()
        self.output_stream = queue.Queue()
        self.logger = Logger()
        self.logger.info(LogComponent.WEBSOCKET, "WebSocket streams initialized")
    
    async def enqueue_input_stream(self, audio_bytes):
        try:
            self.input_stream.put_nowait(audio_bytes)
        except queue.Full:
            self.logger.warning(LogComponent.WEBSOCKET, "Input stream is full")
    
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
            self.logger.warning(LogComponent.WEBSOCKET, "Output stream is full")
    
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
    logger.info(LogComponent.WEBSOCKET, "New WebSocket connection established")
    # first receive is not audio data
    await websocket.receive()
    
    # setup streams and synchronization events
    websocket_streams = WebsocketStreams()
    system_interrupt = threading.Event()
    user_interrupt = threading.Event()

    config = AppConfig(
        aws_region="us-west-2",
        model_id="meta.llama3-1-70b-instruct-v1:0",
        language_code="en-IN"
    )

    # Create conversation controller
    controller = ConversationController(websocket_streams, system_interrupt, user_interrupt, config)
    
    async def read_from_websocket():
        while True:
            data = await websocket.receive()
            await websocket_streams.enqueue_input_stream(data)
            if system_interrupt.is_set():
                logger.info(LogComponent.WEBSOCKET, "Exiting websocket read")
                break
    
    async def write_to_websocket():
        i = 0
        buffer = bytearray()
        while True:
            try:
                data = await websocket_streams.dequeue_output_stream()  
                buffer.extend(data)
                # At 16000 Hz, ensure we are sending 640 bytes per packet
                while len(buffer) > 640:
                    data = buffer[:640]
                    buffer = buffer[640:]   
                    await websocket.send(data)
                    await asyncio.sleep(0.015)
                    i += 1
                if user_interrupt.is_set():
                    logger.info(LogComponent.WEBSOCKET, "User interrupt set, clearing buffer")
                    with websocket_streams.output_stream.mutex:
                        websocket_streams.output_stream.queue.clear()
                        
                if i % 200 == 0:
                    logger.debug(LogComponent.WEBSOCKET, f"Sent {i} packets")
                    await asyncio.sleep(0.5)
                
                if system_interrupt.is_set():
                    logger.info(LogComponent.WEBSOCKET, "Exiting websocket write")
                    break
            except Exception as e:
                logger.error(LogComponent.WEBSOCKET, f"Error in websocket write: {e}")
                break

    def signal_handler():
        logger.warning(LogComponent.SYSTEM, "Received interrupt signal, shutting down...")
        system_interrupt.set()
        user_interrupt.set()  # Unblock any waiting threads
        controller.stop_conversation()
        # Force clear all queues
        with websocket_streams.input_stream.mutex:
            websocket_streams.input_stream.queue.clear()
        with websocket_streams.output_stream.mutex:
            websocket_streams.output_stream.queue.clear()
        
    read_audio_task = asyncio.create_task(read_from_websocket())
    write_audio_task = asyncio.create_task(write_to_websocket())
    
    try:
        # Run conversation controller in executor and gather with websocket tasks
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)

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
    logger.info(LogComponent.WEBSOCKET, "Event received")
    return "Event received", 200

@app.route("/webhooks/answer")
async def answer_call():
    logger.info(LogComponent.WEBSOCKET, "Answering new call")
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