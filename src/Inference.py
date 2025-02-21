import threading
import queue
import asyncio
import json
import boto3
from config import AppConfig, api_request_list
from typing import Optional
from botocore.config import Config
from Logger import Logger, LogComponent
from Metrics import Metrics, MetricType
from BaseThread import BaseThread
import time

class LLMInference(BaseThread):
    def __init__(self, config, user_interrupt, bedrock_context, bedrock_complete, transcribe_complete, transcribe_to_bedrock, bedrock_to_stt):
        super().__init__(name="Inference")
        bedrock_config = Config(
            read_timeout=30,
            retries={'max_attempts': 2}
        )
        self.client = boto3.client('bedrock-runtime', config=bedrock_config, region_name=config.aws_region)
        self.config = config
        self.user_interrupt = user_interrupt
        self.context = bedrock_context
        self.transcribe_complete = transcribe_complete
        self.bedrock_complete = bedrock_complete
        self.transcribe_to_bedrock = transcribe_to_bedrock
        self.bedrock_to_stt = bedrock_to_stt
        self.logger.info(LogComponent.INFERENCE, "LLM Inference initialized")

    def _run_process(self):
        asyncio.run(self._get_response())

    async def _get_response(self):
        try: 
            self.logger.debug(LogComponent.INFERENCE, "Waiting for transcription to complete")
            self.transcribe_complete.wait()
            text = self.transcribe_to_bedrock.get(timeout=0.1)
            self.context.add_user_input(text)
            body = self.define_body(text)
            self.logger.debug(LogComponent.INFERENCE, f"Sending request to Bedrock with text: {body['prompt']}")

            body_json = json.dumps(body)
            model_params = api_request_list[self.config.model_id]
            
            self.metrics.start_metric(MetricType.API_LATENCY, "bedrock_inference")
            self.metrics.start_metric(MetricType.INFERENCE, "bedrock_inference_request_time")
            response = self.client.invoke_model_with_response_stream(
                body=body_json, 
                modelId=model_params['modelId'], 
                accept=model_params['accept'], 
                contentType=model_params['contentType']
            )
            self.metrics.end_metric(MetricType.INFERENCE, "bedrock_inference_request_time")

            full_response = ""
            bedrock_stream = response.get('body')
            first_chunk = True
            
            async for chunk in self.process_stream(bedrock_stream):
                if first_chunk:
                    # Record time to first token
                    self.metrics.record_time_to_first_token("bedrock_inference", "bedrock_first_token")
                    self.metrics.start_metric(MetricType.SPEECH_PROCESSING, "first_chunk_bedrock_to_stt")
                    first_chunk = False
                
                full_response += chunk
                self.bedrock_to_stt.put(chunk)
                if self.user_interrupt.is_set():
                    with self.bedrock_to_stt.mutex:
                        self.bedrock_to_stt.queue.clear()
                    self.logger.warning(LogComponent.INFERENCE, "Inference interrupted by user")
                    break
            
            # Store whatever has already been output
            self.context.add_bedrock_output(full_response)
            self.logger.info(LogComponent.INFERENCE, f"Complete response: {full_response}")
            self.metrics.end_metric(MetricType.API_LATENCY, "bedrock_inference")
                
            self.bedrock_complete.set()
        except queue.Empty:
            self.logger.warning(LogComponent.INFERENCE, "No text received from transcription")
            self.bedrock_complete.set()
        except Exception as e: 
            self.logger.error(LogComponent.INFERENCE, f"Bedrock Error: {e}")
    
    def define_body(self, text):
        body = api_request_list[self.config.model_id]['body']
        body['prompt'] = self.context.get_context()
        return body
  
    async def process_stream(self, bedrock_stream):
        buffer = ''
        punctuation = '.,!?|'
        
        if bedrock_stream:
            for event in bedrock_stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    text = chunk_obj['generation']
                    buffer += text
                
                # Stream out complete sentences on any punctuation
                for punct in punctuation:
                    while punct in buffer:
                        sentence, buffer = buffer.split(punct, 1)
                        yield sentence + punct
                        
            # Yield any remaining text in buffer
            if buffer:
                # Add appropriate punctuation if missing
                if not any(buffer.endswith(p) for p in punctuation):
                    buffer += '.'
                yield buffer

class BedrockContext: 
    def __init__(self, config):
        self.history = []
        self.formatted_context = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{config.system_prompt}<|eot_id|>\n"
        self.logger = Logger()
        self.logger.debug(LogComponent.INFERENCE, "BedrockContext initialized")
  
    def add_user_input(self, user_input):
        self.history.append({"role":"user", "message": user_input})
        
        user_turn = f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>"
        partial_model_tag = f"<|start_header_id|>assistant<|end_header_id|>"
        self.formatted_context += f"{user_turn}\n{partial_model_tag}\n"
    
    def add_bedrock_output(self, bedrock_output):
        self.history.append({"role":"assistant", "message": bedrock_output})
        self.formatted_context += f"{bedrock_output}<|eot_id|>\n"
  
    def get_context(self):
        return self.formatted_context