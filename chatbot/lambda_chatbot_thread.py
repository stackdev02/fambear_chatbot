import os
import sys
import json
import time
from typing import Dict, Any, List

sys.path.append('/opt/openai_package')
sys.path.append('/opt/pinecone_package')
sys.path.append('/opt/requests_package')

import openai
from pinecone import Pinecone

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "fambear"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Add your Telegram bot token here

# Initialize OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)

# System prompt with refined instructions
SYSTEM_PROMPT = """
You are Fambear Assistant, a friendly and professional chatbot designed to help families find the perfect nanny, maid, or tutor 🏡👨‍👩‍👧‍👦. Your goal is to ask relevant questions, understand the family's needs, and recommend suitable candidates.

## **Identity**
- You are an AI assistant for Fambear, a platform connecting families with trusted service providers.
- Your tone is warm, professional, and informative.
- You guide families through the process, ensuring they feel supported.

## **Your Goals**
Engage in a conversation with the family to collect the necessary details:  
1. **Service Type** - What type of service do they need? (Nanny, maid, or tutor) 👶🧹📚  
2. **Budget** - What is their budget? (Hourly or monthly) 💰  
3. **Experience** - How many years of experience should the provider have?  
4. **Language Preferences** - Do they have any language preferences? 🗣  
5. **Job Description** - What are the key responsibilities or requirements? 📝  

Once all required details are provided:  
- Retrieve relevant candidates from the Fambear database using Pinecone and OpenAI embeddings.  
- Present a list of suitable providers with their details, including experience, rates, skills, availability, and profile links.  
- Offer a friendly, professional summary and suggest next steps.  

## **Guidelines & Guardrails**
- Keep the conversation **focused** on helping the family find a service provider.  
- If asked unrelated questions, politely require focus on our goal.  
- Never provide personal opinions or advice outside the Fambear platform.  
- Maintain a **warm and professional tone** 😊, using emoticons sparingly to keep the conversation engaging.  
- **Never mention system processes** (e.g., querying a database or waiting for responses).  
- If no suitable candidates are found, politely inform the family and suggest refining their search criteria.  

Your mission is to make the process seamless and enjoyable, helping families find the best provider for their needs. 🤝🏡  
"""

# Dictionary to store OpenAI thread IDs and their creation timestamps
threads = {}

def send_telegram_message(chat_id: int, text: str) -> None:
    """Send a message to the Telegram bot using the chat_id."""
    import requests

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        print(f"Failed to send message to Telegram: {response.text}")

def cleanup_old_threads():
    """Delete threads older than 1 hour."""
    current_time = time.time()
    expired_threads = []

    for chat_id, thread_info in list(threads.items()):
        thread_id, creation_time = thread_info
        if current_time - creation_time > 3600:  # 1 hour in seconds
            try:
                # Delete the thread using the OpenAI API
                openai.beta.threads.delete(thread_id)
                expired_threads.append(chat_id)
            except Exception as e:
                print(f"Failed to delete thread {thread_id}: {e}")

    # Remove expired threads from the dictionary
    for chat_id in expired_threads:
        del threads[chat_id]

def classify_conversation(messages: List[Dict[str, str]]) -> str:
    """
    Classify the conversation to determine if enough information is available for a service provider search.
    Uses `gpt-4o-mini` for classification.
    """
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    prompt = (
        f"Review the following conversation history and determine if the user has provided enough information "
        f"to perform a service provider search. The required details include:\n"
        f"- What type of service are they looking for? (Nanny, maid, or tutor)\n"
        f"- What is their budget? (Hourly or monthly)\n"
        f"- How many years of experience should the provider have?\n"
        f"- Do they have any language preferences?\n"
        f"- Can they describe the job responsibilities or specific requirements?\n\n"
        f"Conversation History:\n{conversation_history}\n\n"
        f"Decision: If all necessary details are provided, respond with 'ready_for_search'. "
        f"Otherwise, respond with 'need_more_info'."
    )

    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Use gpt-4o-mini for classification
        messages=[
            {"role": "system", "content": "You are a conversation classification assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0  # Use low temperature for deterministic output
    )
    return response.choices[0].message.content.strip()

def get_embeddings(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Generate embeddings for a given text."""
    try:
        response = openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def query_pinecone(query_embedding: List[float], top_k: int = 5, relevance_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Query Pinecone for similar service providers and filter results based on a relevance threshold.
    
    Args:
        query_embedding (List[float]): The embedding vector for the query.
        top_k (int): The number of top matches to retrieve.
        relevance_threshold (float): The minimum similarity score for a match to be considered relevant.
    
    Returns:
        List[Dict[str, Any]]: A list of relevant matches with metadata.
    """
    try:
        # Query Pinecone for matches
        response = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="profiles"
        )
        
        # Filter matches based on the relevance threshold
        relevant_matches = [match for match in response.matches if match.score >= relevance_threshold]
        
        return relevant_matches
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

def format_provider_details(matches: List[Dict[str, Any]]) -> str:
    """Format the retrieved service provider details into a structured string using the provided metadata format."""
    formatted_details = []
    for match in matches:
        metadata = match.metadata
        details = (
            f"**Profile ID:** {metadata.get('profile_id', 'N/A')}\n"
            f"**Profile Link:** {metadata.get('profile_link', 'N/A')}\n"
            f"**User ID:** {metadata.get('user_id', 'N/A')}\n"
            f"**Data:**\n{metadata.get('data', 'No additional data available.')}\n"
        )
        formatted_details.append(details)
    return "\n\n".join(formatted_details)

def generate_combined_response(customer_query: str, provider_details: str) -> str:
    """Generate a professional and engaging response combining the customer query and provider details."""
    prompt = (
        f"A family is looking for a service provider with the following requirements:\n\n"
        f"{customer_query}\n\n"
        f"Below are the best-matching candidates based on their criteria:\n\n"
        f"{provider_details}\n\n"
        f"Please present each candidate's details in the following structured format:\n\n"
        f"Candidate Profile:\n"
        f"🔗 Profile Link: https://--------- \n"
        f"💰 Budget: ---- (monthly) or ---- (hourly)\n"
        f"🎓 Experience: ---- years\n"
        f"🗣 Languages Spoken: -----\n"
        f"📝 Summary: [Brief description tailored to the family's needs]\n\n"
        f"Conclude the response with a warm and professional closing, summarizing the next steps. "
        f"Ensure the response is friendly yet professional, making it easy for the family to review and choose the right provider. "
        f"Use emoticons subtly to maintain a welcoming tone. "
        f"Avoid mentioning system processes like waiting times or database queries."
    )

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def create_structured_query(messages: List[Dict[str, str]]) -> str:
    """Convert conversation history into structured query matching training data format."""
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages if msg['role'] == 'user'])
    
    extraction_prompt = (
        f"Extract service requirements from this conversation and format as:\n\n"
        f"## Description: [Service type, age preferences, job description]\n"
        f"## Hourly Rate: [If mentioned]\n"
        f"## Monthly Rate: [If mentioned]\n"
        f"## Work Days: [Preferred schedule/availability]\n"
        f"## Skills: [Required qualifications]\n\n"
        f"Conversation:\n{conversation_history}\n\n"
        f"Fill only sections with available information. Use exact numbers and terms from the conversation."
    )
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a data extraction specialist."},
            {"role": "user", "content": extraction_prompt}
        ],
        temperature=0.0
    )
    
    return response.choices[0].message.content.strip()

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle POST request from Telegram bot, generate ChatGPT response, and return it."""
    try:

        print(event)

        # Start measuring response time
        start_time = time.time()

        # Clean up old threads before processing the request
        cleanup_old_threads()

        # Extract chat_id and message text from the Telegram event
        chat_id = event["message"]["chat"]["id"]
        user_message = event["message"]["text"]

        # Retrieve or create a thread for the chat_id
        if chat_id not in threads:
            # Create a new thread for the chat_id
            thread = openai.beta.threads.create()
            threads[chat_id] = (thread.id, time.time())  # Store thread ID and creation time
        else:
            # Retrieve the existing thread
            thread_id, _ = threads[chat_id]
            thread = openai.beta.threads.retrieve(thread_id)

        # Add the user's message to the thread
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message
        )

        # Retrieve the conversation history from the thread
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        print(messages)

        # Extract only the string content from the messages
        conversation_history = []
        for msg in messages:
            role = msg.role
            content = ""
            
            # Extract the text content from the message
            for content_block in msg.content:
                if content_block.type == "text":
                    content = content_block.text.value
                    break  # Assuming only one text block per message
            
            conversation_history.append({"role": role, "content": content})

        # Add system prompt to the conversation history
        conversation_history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        # Classify the conversation to determine if enough information is available for a search
        classification = classify_conversation(conversation_history)
        print(classification)

        if classification == "ready_for_search":
            # Extract the latest customer query
            customer_query = create_structured_query(conversation_history)

            # Generate embeddings for the query
            query_embedding = get_embeddings(customer_query)
            if not query_embedding:
                send_telegram_message(chat_id, "Failed to generate embeddings for the query. Please try again.")
                return {
                    "statusCode": 500,
                    "body": json.dumps({"error": "Failed to generate embeddings for the query."})
                }

            # Query Pinecone for matching service providers
            matches = query_pinecone(query_embedding)
            print(matches)
            if not matches:
                send_telegram_message(chat_id, "I couldn't find any candidates that exactly match your requirements at the moment. 😊 However, you can try refining your search criteria (e.g., adjusting the budget, experience level, or availability) or let me know if there's anything else I can help you with!")
                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "reply": "No matches found.",
                        "p_time": round(time.time() - start_time, 2)
                    })
                }

            # Format the retrieved provider details
            provider_details = format_provider_details(matches)

            # Generate a combined response using OpenAI
            combined_response = generate_combined_response(customer_query, provider_details)

            # Send the response to the Telegram bot
            send_telegram_message(chat_id, combined_response)

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "reply": combined_response,
                    "p_time": round(time.time() - start_time, 2)
                })
            }

        else:
            # If more information is needed, ask follow-up questions
            follow_up_prompt = (
                "Based on the conversation history, ask the user for any missing details required for a service provider search. "
                "Be polite and professional."
            )
            print(follow_up_prompt)
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation_history + [{"role": "user", "content": follow_up_prompt}],
                temperature=0.7
            )
            follow_up_reply = response.choices[0].message.content
            print(follow_up_reply)

            # Send the follow-up response to the Telegram bot
            send_telegram_message(chat_id, follow_up_reply)

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "reply": follow_up_reply,
                    "p_time": round(time.time() - start_time, 2)
                })
            }

    except Exception as e:
        send_telegram_message(chat_id, f"An error occurred: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "p_time": round(time.time() - start_time, 2)
            })
        }