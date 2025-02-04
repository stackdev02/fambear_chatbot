import os
import pymysql
import json
import logging
import time
from dotenv import load_dotenv
import openai
from pinecone import Pinecone, ServerlessSpec
from colorama import Fore, Style

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Retrieve credentials
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "fambear"
BATCH_SIZE = 100

# Initialize OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

def ensure_pinecone_index(index_name, dimension=1536, metric="cosine"):
    """Ensure the Pinecone index is deleted and recreated."""
    try:
        existing_indexes = [index['name'] for index in pc.list_indexes().get('indexes', [])]

        if index_name in existing_indexes:
            logger.warning(Fore.YELLOW + f"Index '{index_name}' exists. Deleting it..." + Style.RESET_ALL)
            pc.delete_index(index_name)

            # Polling loop: Wait for deletion to complete
            while True:
                time.sleep(2)  # Check every 2 seconds
                existing_indexes = [index['name'] for index in pc.list_indexes().get('indexes', [])]
                if index_name not in existing_indexes:
                    break

            logger.info(Fore.GREEN + f"Index '{index_name}' deleted successfully." + Style.RESET_ALL)

        logger.info(Fore.CYAN + f"Creating index '{index_name}'..." + Style.RESET_ALL)
        pc.create_index(
            index_name, 
            dimension=dimension, 
            metric=metric, 
            spec=ServerlessSpec(cloud='aws', region=os.getenv("PINECONE_REGION", "us-east-1"))
        )

        # Poll until the new index appears
        while True:
            time.sleep(2)  # Check every 2 seconds
            existing_indexes = [index['name'] for index in pc.list_indexes().get('indexes', [])]
            if index_name in existing_indexes:
                break

        logger.info(Fore.GREEN + f"Index '{index_name}' created successfully." + Style.RESET_ALL)
        return pc.Index(index_name)

    except Exception as e:
        logger.error(Fore.RED + f"Error ensuring Pinecone index: {e}" + Style.RESET_ALL, exc_info=True)
        raise

def fetch_data():
    """Fetch data from MySQL database."""
    try:
        logger.info(Fore.CYAN + "Connecting to database..." + Style.RESET_ALL)
        connection = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
        logger.info(Fore.GREEN + "Connected to database." + Style.RESET_ALL)

        query = """
        SELECT 
            sp.id AS profile_id, sp.user_id, sp.highlights, sp.schedule, sp.note,
            sp.rate_hourly, sp.rate_monthly, sp.work_experience_current,
            sp.work_experience_last_year, sp.work_experience_last_five_year,
            JSON_OBJECTAGG(skill_list.category, skill_list.skills) AS categorized_skills
        FROM service_profiles sp
        LEFT JOIN (
            SELECT 
                sps.profile_id, COALESCE(parent_skills.name, 'Other') AS category,
                JSON_ARRAYAGG(s.name) AS skills
            FROM service_profile_skills sps
            LEFT JOIN skills s ON sps.skill_id = s.id
            LEFT JOIN skills parent_skills ON s.parent_id = parent_skills.id
            GROUP BY sps.profile_id, category
        ) AS skill_list ON sp.id = skill_list.profile_id
        WHERE sp.is_deleted = 0 AND (sp.highlights IS NOT NULL AND sp.highlights <> '')
        GROUP BY sp.id;
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
        connection.close()
        
        return result
    except Exception as e:
        logger.error(Fore.RED + f"Database error: {e}" + Style.RESET_ALL, exc_info=True)
        return []

import json

def process_text(record):
    """Combine key fields into a structured multi-line text format."""

    def format_schedule(schedule_str):
        """Parse JSON schedule and format enabled workdays."""
        try:
            schedule = json.loads(schedule_str) if isinstance(schedule_str, str) else schedule_str
            work_days = [f"- {day}: {info['start']} to {info['end']}" for day, info in schedule.items() if info.get("enable") == "true"]
            return "\n".join(work_days)
        except json.JSONDecodeError:
            return ""

    # Extract fields safely
    highlights = record.get("highlights", "")
    work_experience_current = record.get("work_experience_current", "")
    work_experience_last_year = record.get("work_experience_last_year", "")
    work_experience_last_five_year = record.get("work_experience_last_five_year", "")
    rate_hourly = record.get("rate_hourly", "")
    rate_monthly = record.get("rate_monthly", "")
    schedule = format_schedule(record.get("schedule", "{}"))  # Ensure JSON parsing

    # Process categorized skills
    skills_data = record.get("categorized_skills", {})
    if skills_data is None:
        skills_data = {}
    elif isinstance(skills_data, str):
        try:
            skills_data = json.loads(skills_data)
        except json.JSONDecodeError:
            skills_data = {}

    skills_text = "\n".join([f"- {category}: {', '.join(skills)}" for category, skills in skills_data.items()])

    # Combine all into a multi-line string
    result = (
        f"## Description: {highlights}\n"
        f"## Recent Work Experience: {work_experience_current}\n"
        f"## Previous Work Experience: {work_experience_last_year}\n"
        f"## Other Work Experience: {work_experience_last_five_year}\n"
        f"## Hourly Rate: {rate_hourly}\n"
        f"## Monthly Rate: {rate_monthly}\n"
        f"## Work Days:\n{schedule}\n"
        f"## Skills:\n{skills_text}"
    )

    # logger.info(result)
    return result



def get_embeddings(batch_texts, model="text-embedding-3-small"):
    """Generate embeddings for a batch of texts."""
    try:
        response = openai.embeddings.create(input=batch_texts, model=model)
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        logger.error(Fore.RED + f"Error generating embeddings: {e}" + Style.RESET_ALL, exc_info=True)
        return None

def store_in_pinecone(records, pinecone_index):
    """Store processed records in Pinecone in batches."""
    if not records:
        logger.info(Fore.YELLOW + "No records to store." + Style.RESET_ALL)
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
                    "profile_link": f"https://www.fambear.com/customers/profile/{record["profile_id"]}",
                    "user_id": record["user_id"],
                    "data": text
                }
            }
            for record, vector, text in zip(batch_records, batch_embeddings, batch_texts)
        ]

        if upserts:
            logger.info(Fore.CYAN + "Uploading batch to Pinecone..." + Style.RESET_ALL)
            pinecone_index.upsert(upserts, namespace="service-providers")
            logger.info(Fore.GREEN + f"Stored {len(upserts)} records in Pinecone." + Style.RESET_ALL)

def handler(event=None, context=None):
    start_time = time.time()
    logger.info(Fore.BLUE + "Starting data processing pipeline..." + Style.RESET_ALL)

    pinecone_index = ensure_pinecone_index(INDEX_NAME)
    records = fetch_data()
    store_in_pinecone(records, pinecone_index)

    logger.info(Fore.MAGENTA + f"Total processing time: {time.time() - start_time:.2f} seconds" + Style.RESET_ALL)

if __name__ == "__main__":
    handler()
