import asyncio
import json
import queue
import threading
import signal
from threading import Thread, Event
from queue import Queue
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
from config import AppConfig, api_request_list
from Orchestrate import ConversationController
from Logger import Logger, LogComponent
from Metrics import Metrics, MetricType


app = FastAPI()
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
                await asyncio.sleep(0.05)

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
                await asyncio.sleep(0.05)
    
    
@app.websocket("/socket")
async def handle_websocket(websocket: WebSocket):
    logger.info(LogComponent.WEBSOCKET, "New WebSocket connection established")
    await websocket.accept()
    # first receive is not audio data
    await websocket.receive_json()
    
    # setup streams and synchronization events
    websocket_streams = WebsocketStreams()
    system_interrupt = threading.Event()
    user_interrupt = threading.Event()
    metrics = Metrics()
    read_audio_task = None
    write_audio_task = None

    config = AppConfig(
        aws_region="us-west-2",
        model_id="meta.llama3-1-70b-instruct-v1:0",
        language_code="en-US"
    )

    # Create conversation controller
    controller = ConversationController(websocket_streams, system_interrupt, user_interrupt, config)
    
    async def read_from_websocket():
        while True:
            try:
                data = await websocket.receive_bytes()
                await websocket_streams.enqueue_input_stream(data)
                if system_interrupt.is_set():
                    logger.info(LogComponent.WEBSOCKET, "Exiting websocket read")
                    break
            except Exception as e:
                logger.error(LogComponent.WEBSOCKET, f"Error in websocket read: {e}")
                break
    
    async def write_to_websocket():
        first_chunk = True
        i = 1
        buffer = bytearray()
        while True:
            try:
                data = await websocket_streams.dequeue_output_stream()  
                buffer.extend(data)
                if first_chunk:
                    metrics.end_metric(MetricType.SPEECH_PROCESSING, "first_chunk_to_output_stream")
                    metrics.end_metric(MetricType.THREAD_LIFETIME, "*****response_time_from_silence*****")
                    first_chunk = False

                # At 16000 Hz, ensure we are sending 640 bytes per packet
                while len(buffer) > 640:
                    data = buffer[:640]
                    buffer = buffer[640:]   
                    await websocket.send_bytes(bytes(data))
                    await asyncio.sleep(0.015)
                    i += 1

                if user_interrupt.is_set():
                    logger.info(LogComponent.WEBSOCKET, "User interrupt set, clearing buffer")
                    with websocket_streams.output_stream.mutex:
                        websocket_streams.output_stream.queue.clear()
                        buffer = bytearray()
                        
                if system_interrupt.is_set():
                    logger.info(LogComponent.WEBSOCKET, "Exiting websocket write")
                    break
                        
                if i % 200 == 0:
                    logger.debug(LogComponent.WEBSOCKET, f"Sent {i} packets")
                    await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(LogComponent.WEBSOCKET, f"Error in websocket write: {e}")
                break

    # Move signal_handler inside to access all variables
    def signal_handler():
        nonlocal read_audio_task, write_audio_task
        logger.warning(LogComponent.SYSTEM, "Received interrupt signal, shutting down...")
        try:
            # Set system interrupt to break loops
            system_interrupt.set()
            
            # First stop the conversation controller
            controller.stop_conversation()
            
            # Then clear queues
            with websocket_streams.input_stream.mutex:
                websocket_streams.input_stream.queue.clear()
            with websocket_streams.output_stream.mutex:
                websocket_streams.output_stream.queue.clear()
            
            # Cancel the websocket tasks
            if read_audio_task and not read_audio_task.done():
                read_audio_task.cancel()
            if write_audio_task and not write_audio_task.done():
                print("Cancelling write_audio_task")
                write_audio_task.cancel()
            
            # Finally stop the event loop
            loop = asyncio.get_running_loop()
            loop.create_task(websocket.close())
            loop.stop()
        except Exception as e:
            logger.error(LogComponent.SYSTEM, f"Error in shutdown: {e}")

    try:
        # Run conversation controller in executor and gather with websocket tasks
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)

        read_audio_task = asyncio.create_task(read_from_websocket())
        write_audio_task = asyncio.create_task(write_to_websocket())

        await asyncio.gather(
            read_audio_task,
            write_audio_task,
            loop.run_in_executor(None, controller.start_conversation)
        )
    except Exception as e:
        logger.error(LogComponent.WEBSOCKET, f"Error in websocket handler: {e}")
    finally:
        if read_audio_task:
            read_audio_task.cancel()
        if write_audio_task:
            write_audio_task.cancel()
        controller.stop_conversation()
        await websocket.close()
    
    
@app.post("/webhooks/events")
@app.get("/webhooks/events")
async def events():
    logger.info(LogComponent.WEBSOCKET, "Event received")
    return "Event received", 200

@app.get("/webhooks/answer")
async def answer_call(request: Request):
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
                    "uri": f"wss://{request.headers['host']}/socket",
                    "content-type": "audio/l16;rate=16000",
                }
            ],
        },
    ]

    return JSONResponse(content=ncco)
    