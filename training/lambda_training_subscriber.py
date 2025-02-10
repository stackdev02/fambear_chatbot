import os
import sys
import json
from typing import Dict, Any

sys.path.append('/opt/openai_package')
sys.path.append('/opt/pinecone_package')


import openai
from pinecone import Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "fambear"
NAMESPACE = "profiles"
BATCH_SIZE = 100

# Initialize OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

def check_namespace_exists(index_name, namespace):
    """Check if the namespace exists in the index."""
    index = pc.Index(index_name)
    return index.describe_index_stats(namespace=namespace)

def process_text(record):
    """Combine key fields into a structured multi-line text format."""
    def format_schedule(schedule_str):
        try:
            schedule = json.loads(schedule_str) if isinstance(schedule_str, str) else schedule_str
            work_days = [f"- {day}: {info['start']} to {info['end']}" for day, info in schedule.items() if info.get("enable") == "true"]
            return "\n".join(work_days)
        except json.JSONDecodeError:
            return ""

    highlights = record.get("highlights", "")
    work_experience_current = record.get("work_experience_current", "")
    work_experience_last_year = record.get("work_experience_last_year", "")
    work_experience_last_five_year = record.get("work_experience_last_five_year", "")
    rate_hourly = record.get("rate_hourly", "")
    rate_monthly = record.get("rate_monthly", "")
    schedule = format_schedule(record.get("schedule", "{}"))
    
    skills_data = record.get("categorized_skills", {})
    if isinstance(skills_data, str):
        try:
            skills_data = json.loads(skills_data)
        except json.JSONDecodeError:
            skills_data = {}

    skills_text = "\n".join([f"- {category}: {', '.join(skills)}" for category, skills in skills_data.items()])

    return (
        f"## Description: {highlights}\n"
        f"## Recent Work Experience: {work_experience_current}\n"
        f"## Previous Work Experience: {work_experience_last_year}\n"
        f"## Other Work Experience: {work_experience_last_five_year}\n"
        f"## Hourly Rate: {rate_hourly}\n"
        f"## Monthly Rate: {rate_monthly}\n"
        f"## Work Days:\n{schedule}\n"
        f"## Skills:\n{skills_text}"
    )

def get_embeddings(batch_texts, model="text-embedding-3-small"):
    try:
        response = openai.embeddings.create(input=batch_texts, model=model)
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def store_in_pinecone(records, pinecone_index, namespace):
    if not records:
        print("No records to store.")
        return
    
    texts = [process_text(record) for record in records]
    for i in range(0, len(records), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_embeddings = get_embeddings(batch_texts)
        batch_records = records[i:i + BATCH_SIZE]

        upserts = [
            {
                "id": str(record["profile_id"]),
                "values": vector,
                "metadata": {
                    "profile_id": record["profile_id"],
                    "profile_link": f"https://www.fambear.com/customers/profile/{record['profile_id']}",
                    "user_id": record["user_id"],
                    "data": text
                }
            }
            for record, vector, text in zip(batch_records, batch_embeddings, batch_texts)
        ]

        if upserts:
            print("Uploading batch to Pinecone...")
            pinecone_index.upsert(upserts, namespace=namespace)
            print(f"Stored {len(upserts)} records in Pinecone.")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle SNS messages containing batches of records to process."""
    try:
        # Initialize Pinecone index
        if not check_namespace_exists(INDEX_NAME, "profiles"):
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'Namespace "profiles" does not exist in index "{INDEX_NAME}".'
                })
            }

        pinecone_index = pc.Index(INDEX_NAME)

        # Process SNS message
        for record in event['Records']:
            message = json.loads(record['Sns']['Message'])
            batch_records = message['records']
            
            # Process records based on their type (update or delete)
            updates = []
            deletes = []
            
            for record in batch_records:
                if record['type'] == 'update':
                    updates.append(record)
                elif record['type'] == 'delete':
                    deletes.append(record['profile_id'])
            
            # Handle updates
            if updates:
                store_in_pinecone(updates, pinecone_index, NAMESPACE)
            
            # Handle deletes
            if deletes:
                print("Deleting records from Pinecone...")
                pinecone_index.delete(ids=deletes, namespace=NAMESPACE)
                print(f"Deleted {len(deletes)} records from Pinecone")
        


        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Successfully processed batch: Stored {len(updates)}, Deleted {len(deletes)}'
            })

        }
        
    except Exception as e:
        print(f"Error in lambda_handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

