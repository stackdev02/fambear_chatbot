import os
import sys
import json

sys.path.append('/opt/boto3_package')
sys.path.append('/opt/pymysql_package')

import boto3
import pymysql
from typing import List, Dict, Any

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

# SNS configuration
sns_client = boto3.client('sns')
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN")
BATCH_SIZE = 50

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
        GROUP BY sp.id;
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
        # Fetch data from database
        records = fetch_data()
        print(f"Fetched {len(records)} records from database")

        # Split records into batches and publish to SNS
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            
            # Publish batch to SNS
            response = sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=json.dumps({
                    # 'batch_number': i // BATCH_SIZE + 1,
                    # 'total_batches': (len(records) + BATCH_SIZE - 1) // BATCH_SIZE,
                    'records': batch
                })
            )
            print(f"Published batch {i} to SNS: {response['MessageId']}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Successfully processed {len(records)} records',
                'total_batches': (len(records)+BATCH_SIZE)/BATCH_SIZE - 1
            })
        }

    except Exception as e:
        print(f"Error in lambda_handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        } 