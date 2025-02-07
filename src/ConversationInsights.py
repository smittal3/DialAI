import json
from datetime import datetime
import boto3
from Logger import Logger, LogComponent
from Database import Database
from typing import List, Dict
from config import AppConfig
import os
import time
from botocore.exceptions import ClientError

class ConversationInsights:
    def __init__(self, config):
        self.config = config
        self.database = Database()
        self.bedrock = boto3.client('bedrock-runtime', region_name=config.aws_region)
        self.base_delay = 1  # Base delay in seconds
        self.max_retries = 5  # Maximum number of retries
        
    def _invoke_model_with_retry(self, prompt: str, max_gen_len: int = 250) -> str:
        """Invoke the model with retry logic for throttling."""
        delay = self.base_delay
        retries = 0
        
        while retries < self.max_retries:
            try:
                response = self.bedrock.invoke_model(
                    modelId=self.config.model_id,
                    body=json.dumps({
                        "prompt": prompt,
                        "max_gen_len": max_gen_len,
                        "temperature": 0.1,
                        "top_p": 0.9
                    })
                )
                return response['body'].read().decode()
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ThrottlingException':
                    retries += 1
                    if retries == self.max_retries:
                        print(f"Max retries ({self.max_retries}) reached. Giving up.")
                        raise e
                        
                    wait_time = delay * (2 ** (retries - 1))  # Exponential backoff
                    print(f"Request throttled. Waiting {wait_time} seconds before retry {retries}/{self.max_retries}")
                    time.sleep(wait_time)
                else:
                    # If it's not a throttling error, raise it immediately
                    raise e
        
        raise Exception("Should not reach here")
        
    def _get_conversation_sentiment(self, conversation_text: str, conversation_id: str) -> Dict:
        """Analyze sentiment for a single conversation with specific questions."""
        print(f"\nAnalyzing sentiment for conversation {conversation_id}")
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a conversation analyzer and you follow rules. You are analyzing conversations that represent
        a call between a representative of an NGO and a beneficiary. The call is surveying the beneficiary.
        Analyze the following conversation and answer these specific questions:
        1. Is the overall sentiment positive or negative? Answer with only one word: 'positive' or 'negative'
        2. Pick max one quote of maximum 15 words that demonstrates this sentiment. Make sure this is a direct quote, do not change the words.

        Respond only with the one word, and on the next line the quote. 
        Do not include any other text or comments.
        
        Here is the conversation:
        {conversation_text}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        print("Requesting sentiment analysis...")
        response_text = self._invoke_model_with_retry(prompt)
        print(f"Response received: {response_text}")
        
        # Parse the JSON response
        response_json = json.loads(response_text)
        generation_text = response_json["generation"]
        print(f"Generation text: {generation_text}")
        
        # Split the generation text into lines
        lines = [line.strip() for line in generation_text.split('\n') if line.strip()]
        print(f"Parsed lines: {lines}")
        
        sentiment = lines[0].lower() if lines else "negative"  # Default to negative if no clear sentiment
        quotes = lines[1:2] if len(lines) > 1 else []  # Take only the second line if it exists
        
        return {
            "conversation_id": conversation_id,
            "sentiment": "positive" if "positive" in sentiment else "negative",
            "supporting_quotes": quotes
        }
    
    def _get_conversation_flags(self, conversation_text: str, conversation_id: str) -> Dict:
        """Identify flags for a single conversation with specific questions."""
        print(f"\nChecking flags for conversation {conversation_id}")
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a conversation analyzer and you follow rules. You are analyzing conversations that represent
        a call between a representative of an NGO and a beneficiary. The call is surveying the beneficiary.
        Analyze this conversation and answer these questions in order:
        1. Should this conversation be flagged for attention? Answer only 'yes' or 'no'
        2. If yes, what is the main reason for flagging? One brief sentence.
        3. If yes, what is the severity? Answer with only: 'high', 'medium', or 'low'
        4. If yes, what type of issue is it? Answer with only: 'technical', 'fraud', or 'urgent'
        5. If yes, pick max one quote of maximum 15 words that demonstrates this sentiment. Make sure this is a direct quote, do not change the words.

        Respond only with the answers in order, one per line.
        Do not include any other text or comments.
        
        Here is the conversation:
        {conversation_text}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        print("Requesting flag analysis...")
        response_text = self._invoke_model_with_retry(prompt)
        print(f"Response received: {response_text}")
        
        try:
            # Parse the JSON response
            response_json = json.loads(response_text)
            generation_text = response_json["generation"].strip()
            print(f"Generation text: {generation_text}")
            
            # Split the generation text into lines and clean them
            lines = [line.strip() for line in generation_text.split('\n') if line.strip()]
            print(f"Parsed lines: {lines}")
            
            if not lines:
                print("No lines found in response")
                return None
                
            should_flag = 'yes' in lines[0].lower()
            print(f"Should flag: {should_flag}")
            
            if not should_flag:
                return None
                
            # Clean up the responses by removing any numbering and quotes
            flag_reason = lines[1].split('.')[-1].strip() if len(lines) > 1 else "Unknown reason"
            severity = lines[2].lower().split('.')[-1].strip() if len(lines) > 2 else "low"
            issue_type = lines[3].lower().split('.')[-1].strip() if len(lines) > 3 else "complaint"
            
            # Clean up quote by removing numbering, quotes, and extra whitespace
            quotes = []
            if len(lines) > 4:
                quote = lines[4]
                # Remove numbering (e.g., "5.")
                if '.' in quote:
                    quote = quote.split('.', 1)[1]
                # Remove surrounding quotes if present
                quote = quote.strip().strip('"').strip()
                quotes = [quote] if quote else []
                
            return {
                "conversation_id": conversation_id,
                "flag_reason": flag_reason,
                "severity": severity,
                "issue_type": issue_type,
                "relevant_quotes": quotes
            }
        except Exception as e:
            print(f"Error parsing flag response: {str(e)}")
            return None
    
    def _get_conversation_feedback(self, conversation_text: str, conversation_id: str) -> Dict:
        """Extract feedback from a single conversation with specific questions."""
        print(f"\nExtracting feedback from conversation {conversation_id}")
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a conversation analyzer and you follow rules. You are analyzing conversations that represent
        a call between a representative of an NGO and a beneficiary. The call is surveying the beneficiary.
        Analyze this conversation and answer these questions in order:
        1. Does this conversation contain any feedback or suggestions? Answer only 'yes' or 'no'
        2. If yes, what type of feedback is it? Answer with only: 'suggestion' or 'pain_point'
        3. If yes, rate the impact as: 'high', 'medium', or 'low'
        4. If yes, pick max one quote of maximum 15 words that demonstrates this sentiment. Make sure this is a direct quote, do not change the words.

        Respond only with the answers in order, one per line.
        Do not include any other text or comments.
        
        Here is the conversation:
        {conversation_text}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        print("Requesting feedback analysis...")
        response_text = self._invoke_model_with_retry(prompt)
        print(f"Response received: {response_text}")
        
        try:
            # Parse the JSON response
            response_json = json.loads(response_text)
            generation_text = response_json["generation"].strip()
            print(f"Generation text: {generation_text}")
            
            # Split the generation text into lines and clean them
            lines = [line.strip() for line in generation_text.split('\n') if line.strip()]
            print(f"Parsed lines: {lines}")
            
            if not lines:
                print("No lines found in response")
                return None
                
            has_feedback = 'yes' in lines[0].lower()
            print(f"Has feedback: {has_feedback}")
            
            if not has_feedback:
                return None
                
            # Clean up the responses by removing any numbering and quotes
            feedback_type = lines[1].lower().split('.')[-1].strip() if len(lines) > 1 else "suggestion"
            impact = lines[2].lower().split('.')[-1].strip() if len(lines) > 2 else "low"
            
            # Clean up quote by removing numbering, quotes, and extra whitespace
            quotes = []
            if len(lines) > 3:
                quote = lines[3]
                # Remove numbering (e.g., "4.")
                if '.' in quote:
                    quote = quote.split('.', 1)[1]
                # Remove surrounding quotes if present
                quote = quote.strip().strip('"').strip()
                quotes = [quote] if quote else []
                
            return {
                "conversation_id": conversation_id,
                "feedback_type": feedback_type,
                "impact": impact,
                "supporting_quotes": quotes
            }
        except Exception as e:
            print(f"Error parsing feedback response: {str(e)}")
            return None

    def analyze_conversation(self, conversation: tuple) -> Dict:
        """Analyze a single conversation comprehensively."""
        conv_id = conversation[0]
        print(f"\n=== Analyzing Conversation {conv_id} ===")
        
        try:
            # Prepare conversation text
            history = conversation[4]
            if isinstance(history, str):
                history = json.loads(history)
                print("Parsed JSON history")
            
            # Map roles for clarity
            role_mapping = {
                "assistant": "Agent",
                "user": "Customer"
            }
            
            conversation_text = "\n".join([
                f"{role_mapping.get(turn['role'], turn['role'])}: {turn['message']}" 
                for turn in history
            ])
            print(f"Conversation length: {len(conversation_text)} characters")
            
            # Analyze each aspect with better error handling
            sentiment_result = None
            flag_result = None
            feedback_result = None
            
            try:
                sentiment_result = self._get_conversation_sentiment(conversation_text, conv_id)
            except Exception as e:
                print(f"Error in sentiment analysis: {str(e)}")
                
            try:
                flag_result = self._get_conversation_flags(conversation_text, conv_id)
            except Exception as e:
                print(f"Error in flag analysis: {str(e)}")
                
            try:
                feedback_result = self._get_conversation_feedback(conversation_text, conv_id)
            except Exception as e:
                print(f"Error in feedback analysis: {str(e)}")
            
            # Compile results
            return {
                "conversation_data": {
                    "id": conv_id,
                    "start_time": conversation[1].isoformat(),
                    "end_time": conversation[2].isoformat(),
                    "duration_seconds": conversation[3],
                    "transcript": conversation_text
                },
                "insights": {
                    "sentiment": sentiment_result,
                    "flags": flag_result,
                    "feedback": feedback_result
                }
            }
        except Exception as e:
            print(f"Error analyzing conversation {conv_id}: {str(e)}")
            return None

    def analyze_all_conversations(self) -> Dict:
        """Analyze all conversations one at a time."""
        print("\n=== Starting Full Analysis ===")
        conversations = self.database.get_all_conversations()
        print(f"Total conversations to analyze: {len(conversations)}")
            
        all_results = {
            "conversations": {},  # Will store all conversation data and insights
            "summary": {
                "total_analyzed": 0,
                "positive_sentiment": 0,
                "negative_sentiment": 0,
                "flagged": 0,
                "with_feedback": 0
            }
        }
        
        for i, conversation in enumerate(conversations, 1):
            print(f"\nProcessing conversation {i} of {len(conversations)}")
            print(f"Conversation: {i}")
            result = self.analyze_conversation(conversation)
            if result is None:
                print(f"Skipping conversation {i} due to analysis error")
                continue
                
            conv_id = result["conversation_data"]["id"]
            
            # Store results in a streamlined structure
            all_results["conversations"][conv_id] = {
                "metadata": {
                    "start_time": result["conversation_data"]["start_time"],
                    "end_time": result["conversation_data"]["end_time"],
                    "duration_seconds": result["conversation_data"]["duration_seconds"]
                },
                "transcript": result["conversation_data"]["transcript"],
                "sentiment": result["insights"]["sentiment"],
                "flags": result["insights"]["flags"],
                "feedback": result["insights"]["feedback"]
            }
            
            # Update summary statistics
            all_results["summary"]["total_analyzed"] += 1
            if result["insights"]["sentiment"]:
                if result["insights"]["sentiment"]["sentiment"] == "positive":
                    all_results["summary"]["positive_sentiment"] += 1
                else:
                    all_results["summary"]["negative_sentiment"] += 1
            if result["insights"]["flags"]:
                all_results["summary"]["flagged"] += 1
            if result["insights"]["feedback"]:
                all_results["summary"]["with_feedback"] += 1
            
            print(f"Conversation {conv_id} analysis complete")
            print("Analysis results:")
            print(f"- Sentiment: {result['insights']['sentiment']['sentiment'] if result['insights']['sentiment'] else 'unknown'}")
            print(f"- Flags: {'Yes' if result['insights']['flags'] else 'No'}")
            print(f"- Feedback: {'Yes' if result['insights']['feedback'] else 'No'}")
        
        print("\n=== Analysis Complete ===")
        print("Summary:")
        print(f"- Total conversations processed: {all_results['summary']['total_analyzed']}")
        print(f"- Positive sentiment: {all_results['summary']['positive_sentiment']}")
        print(f"- Negative sentiment: {all_results['summary']['negative_sentiment']}")
        print(f"- Flagged conversations: {all_results['summary']['flagged']}")
        print(f"- Conversations with feedback: {all_results['summary']['with_feedback']}")
        
        return all_results

    def close(self):
        """Close database connection."""
        self.database.close()


def main():
    try:
        # Initialize config
        config = AppConfig(
            aws_region="us-west-2",
            model_id="meta.llama3-1-70b-instruct-v1:0",
            language_code="en-US"
        )
        
        # Create insights directory if it doesn't exist
        output_dir = "insights"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"conversation_insights_{timestamp}.json")
        
        # Run analysis
        insights = ConversationInsights(config)
        results = insights.analyze_all_conversations()
        
            # Write results to file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
                
        print(f"\nResults written to: {output_file}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise e
    finally:
        insights.close()


if __name__ == "__main__":
    main() 
