# DialAI
DialAI allows you to have a conversation with your favorite LLM (Large Language Model) through voice. This repository implements a conversational flow that integrates with AWS services such as Bedrock, Polly, and Transcribe.

The audio data is recorded in batches and streamed to the Transcribe client for real-time transcription. The system includes silence detection and can handle interrupts or reset the conversation. Asynchronous programming is used to ensure a responsive and smooth user experience.

# Running the app

Before running the application, you'll need to set up your AWS credentials as environment variables. It is recommended to do this in an env.sh file (which is part of .gitignore to ensure sensitive information is not committed).

```shell
export AWS_ACCESS_KEY_ID=<...>
export AWS_SECRET_ACCESS_KEY=<...>
export AWS_DEFAULT_REGION=<...> # Optional, defaults to us-east-1
```

Once the environment variables are set, install the required dependencies:

```shell
pip3 install -r ./requirements.txt
```

Next, you can run the application using:
```shell
sudo -E python3 main.py
```
Note: sudo privileges are required to detect interrupts from the keyboard. If you don't want to use sudo, you can run the app without it, though interrupt detection may not work.