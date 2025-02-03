import os
import json
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def lambda_handler(event, context):
    """Handle POST request from Telegram bot, generate ChatGPT response, and return it."""
    
    try:
        # Parse request body (AWS Lambda sends event as a string)
        body = json.loads(event["body"])

        # Extract conversation history from request
        messages = body.get("history", [])

        if not messages:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing conversation history"})
            }

        # Generate response using OpenAI
        response = openai.chat.completions.create(
            model="gpt-4",  # Change to "gpt-3.5-turbo" if needed
            messages=messages,
            temperature=0.7
        )

        chatgpt_reply = response.choices[0].message.content

        # Return response in JSON format
        return {
            "statusCode": 200,
            "body": json.dumps({"reply": chatgpt_reply})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
