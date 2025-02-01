import threading
import queue
import asyncio
import json
import boto3
from config import AppConfig, api_request_list
from typing import Optional
from botocore.config import Config

class LLMInference:
    def __init__(self, config, user_interrupt, bedrock_context, bedrock_complete, transcribe_to_bedrock, bedrock_to_stt):
        bedrock_config = Config(
        read_timeout=30,
        retries={'max_attempts': 2}
        )
        self.client = boto3.client('bedrock-runtime', config=bedrock_config, region_name=config.aws_region)
        self.config = config
        self.user_interrupt = user_interrupt
        self.context = bedrock_context
        self.bedrock_complete = bedrock_complete
        self.transcribe_to_bedrock = transcribe_to_bedrock
        self.bedrock_to_stt = bedrock_to_stt
        self.thread: Optional[threading.Thread] = None
        self.is_running = True

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        def run_async_inference():
            asyncio.run(self._get_response())

        self.thread = threading.Thread(target=run_async_inference)
        self.thread.start()

    def stop(self):
        if self.thread is not None:
            print("Stopping bedrock thread")
            self.is_running = False
            self.thread.join()
            self.thread = None

    async def _get_response(self):
        try: 
            text = self.transcribe_to_bedrock.get(timeout=3)
            self.context.add_user_input(text)
            body = self.define_body(text)

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
                self.bedrock_to_stt.put(chunk)
                if self.user_interrupt.is_set():
                    with self.bedrock_to_stt.mutex:
                        self.bedrock_to_stt.queue.clear()
                    print("\nInference interrupted!")
                    break
            
            # Store whatever has already been output
            self.context.add_bedrock_output(full_response)
            print(f"Bedrock output: {full_response}")
                
            self.bedrock_complete.set()
        except queue.Empty:
            print("No text to send to bedrock")
            self.bedrock_complete.set()
                
        except Exception as e: 
            print(f"Bedrock Error {e}")
            return "no response, error"

    
    def define_body(self, text):
        body = api_request_list[self.config.model_id]['body']
        body['prompt'] = self.context.get_context()
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