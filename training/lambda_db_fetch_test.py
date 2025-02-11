import os
import json
from dotenv import load_dotenv

import boto3
import pymysql
from typing import List, Dict, Any


load_dotenv()

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

# SNS configuration
try:
    aws_region = os.getenv('AWS_REGION')
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not all([aws_region, aws_access_key, aws_secret_key]):
        raise ValueError("Missing required AWS credentials or region")
    
    sns_client = boto3.client(
        'sns',
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    print(f"Successfully initialized SNS client with region: {aws_region}")
except Exception as e:
    print(f"Error initializing SNS client: {e}")
    sns_client = None

SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN")
BATCH_SIZE = 10

def fetch_data():
    """Fetch data from MySQL database."""
    try:
        print("Connecting to database...")
        connection = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
        print("Connected to database.")

        query = """
        SELECT 
            'update' AS type, sp.id AS profile_id, sp.user_id, sp.highlights, sp.schedule, sp.note,
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
        GROUP BY sp.id
        LIMIT 40;
        """
        
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
        connection.close()
        
        return result
    except Exception as e:
        print(f"Database error: {e}")
        return []

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        if sns_client is None:
            raise RuntimeError("SNS client was not properly initialized")
            
        # Fetch data from database
        records = fetch_data()
        print(f"Fetched {len(records)} records from database")

        if not records:
            print("No records to process")
            return

        # Split records into batches and publish to SNS
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            try:
                # Publish batch to SNS
                response = sns_client.publish(
                    TopicArn=SNS_TOPIC_ARN,
                    Message=json.dumps({
                        'records': batch
                    })
                )
                print(f"Published batch {i//BATCH_SIZE + 1} to SNS: {response['MessageId']}")
            except Exception as e:
                print(f"Error publishing batch {i//BATCH_SIZE + 1} to SNS: {e}")
                raise

        print(f'Successfully processed {len(records)} records')

    except Exception as e:
        print(f"Error in lambda_handler: {e}")
        raise

lambda_handler([], "")