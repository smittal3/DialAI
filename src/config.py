from dataclasses import dataclass, field

api_request_list = {
    'meta.llama3-1-70b-instruct-v1:0': {
        "modelId": "meta.llama3-1-70b-instruct-v1:0",
        "contentType": "application/json",
        "accept": "*/*",
        "latency":"optimized",
        "body": {"prompt":"",
            "max_gen_len":250,
            "temperature":0.1,
            "top_p":0.9,
            }           
    },
    'meta.llama3-3-70b-instruct-v1:0': {
        "modelId": "meta.llama3-3-70b-instruct-v1:0",
        "contentType": "application/json",
        "accept": "*/*",
        "latency":"optimized",
        "body": {"prompt":"",
            "max_gen_len":250,
            "temperature":0.1,
            "top_p":0.9,
            }           
    },
}

def get_model_ids():
    return list(api_request_list.keys())

system_prompt = "Hi, you are a helpful assistant. Keep your responses concise and under 20 words. Be conversational."

@dataclass
class AppConfig:
    """Configuration for the voice chat application"""
    aws_region: str
    model_id: str
    language_code: str
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    logging_verbosity: int = 2
    log_to_stdout: bool = False
    system_prompt: str = system_prompt
    polly: dict = field(default_factory=lambda: {
        'engine' : 'neural', 
        'language' : 'en-US', 
        'voice' : 'Danielle', 
        'outputFormat' : 'pcm'
    })