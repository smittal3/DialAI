from dataclasses import dataclass

api_request_list = {
    'meta.llama3-1-70b-instruct-v1:0': {
        "modelId": "meta.llama3-1-70b-instruct-v1:0",
        "contentType": "application/json",
        "accept": "*/*",
        "body": {"prompt":"You are my friend and You'll watch me play league of legends and teach me how to play it better. You are a good player, you don't like long conversations, you can chat like human.",
            "max_gen_len":250,
            "temperature":0.1,
            "top_p":0.9,
            }           
    },
}

def get_model_ids():
    return list(api_request_list.keys())


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