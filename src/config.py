from dataclasses import dataclass, field

api_request_list = {
    'us.anthropic.claude-3-5-haiku-20241022-v1:0': {
        "modelId": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "system": "", 
            "max_tokens": 200,
            "top_k": 250,
            "stopSequences": [],
            "temperature": 1,
            "top_p": 0.999,
            "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": "hello world"
                }
                ]
            }
            ]
        }
    },
    'meta.llama3-2-3b-instruct-v1:0': {
        "modelId": "meta.llama3-2-3b-instruct-v1:0",
        "contentType": "application/json",
        "accept": "*/*",
        "latency":"optimized",
        "body": {"prompt":"",
            "max_gen_len":250,
            "temperature":0.1,
            "top_p":0.9,
        }
    },
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

system_prompt = (
    "You are Aditi, a representative of FundANeed Foundation, an NGO in India. You are having a phone conversation "
    "with a beneficiary of our cash transfer program. Your role is to gather feedback by asking these 3 questions:"
    "1. Is this Rajeev Singh?"
    "2. Have you received 10,000 rupees from the FundANeed Foundation?"
    "3. How has this money helped you?"
    "Guidelines:"
    "- Start by introducing yourself and asking which language they prefer (Hindi or English)"
    "- Once they respond, stick to ONLY that language for the entire conversation"
    "- DO NOT repeat your responses in multiple languages"
    "- Be warm and empathetic while maintaining professional boundaries"
    "- Keep responses concise and focused on getting specific feedback"
    "- If conversation strays, gently return to the next question in the sequence"
    "- End by thanking them for their feedback"
)


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
        'language' : 'en-IN', 
        'voice' : 'Kajal', 
        'outputFormat' : 'pcm'
    })