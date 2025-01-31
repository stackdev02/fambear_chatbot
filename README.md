# Fambear Chatbot with AWS Lambda

This project is an AI-powered chatbot that uses Retrieval-Augmented Generation (RAG) to generate contextually accurate answers. The chatbot is built using AWS Lambda functions, and it integrates various tools and services, including MySQL for data storage, OpenAI’s embedding model for vector representation, and Pinecone for vector storage and retrieval. The chatbot leverages prompt engineering to enhance its response generation.

---

## Features
- **Data Fetching**: Retrieve datasets from a MySQL database.
- **Data Preprocessing**: Clean and prepare the dataset for embedding.
- **Embedding Generation**: Use OpenAI's `text-embedding-3-small` model to generate vector embeddings.
- **Vector Storage**: Store embeddings in Pinecone for efficient similarity search.
- **Answer Generation**: Use a RAG-based approach combined with prompt engineering to generate accurate and context-aware responses.
- **Serverless**: Built entirely using AWS Lambda functions for scalability and cost efficiency.

---

## Architecture Overview
1. **Data Ingestion**: Data is fetched from a MySQL database using an AWS Lambda function.
2. **Preprocessing**: The dataset is cleaned, tokenized, and prepared for embedding.
3. **Embedding Creation**: OpenAI's `text-embedding-3-small` model generates vector embeddings of the data.
4. **Vector Storage**: The embeddings are stored in Pinecone, a vector database, for efficient retrieval.
5. **Query Handling**:
   - User input is processed and transformed into an embedding.
   - The embedding is used to query Pinecone for relevant context.
   - The retrieved context is combined with the user query using RAG and prompt engineering.
   - The final response is generated and sent to the user.

---

## Prerequisites
Make sure you have the following prerequisites installed and set up:
1. **AWS Account**: To deploy Lambda functions.
2. **Python 3.9 or Later**: For writing and testing the code.
3. **MySQL Database**: For storing the dataset.
4. **Pinecone Account**: For storing and retrieving vector embeddings.
5. **OpenAI API Key**: To access the `text-embedding-3-small` model.

---

## Technologies Used
- **AWS Lambda**: Serverless computing for handling the chatbot's backend logic.
- **MySQL**: Database for storing the chatbot's dataset.
- **Pinecone**: Vector database for storing and retrieving embeddings.
- **OpenAI API**: For embedding generation and prompt-based answer generation.
- **Python**: Programming language for building the chatbot logic.
- **RAG (Retrieval-Augmented Generation)**: Approach for combining retrieval and generation to produce accurate responses.

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your_username/ai-rag-chatbot.git
cd ai-rag-chatbot
```

### 2. Install Dependencies
Set up a virtual environment and install the required Python packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory and add the following environment variables:
```env
MYSQL_HOST=your_mysql_host
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=your_database_name
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
```

### 4. Deploy AWS Lambda Functions
Package the code and deploy it to AWS Lambda. You can use the AWS CLI or a deployment tool like [Serverless Framework](https://www.serverless.com/).

Example using AWS CLI:
```bash
zip -r function.zip .
aws lambda create-function \
    --function-name Fambear-Chatbot \
    --runtime python3.12 \
    --role arn:aws:iam::your_account_id:role/your_lambda_role \
    --handler app.lambda_handler \
    --zip-file fileb://function.zip
```

---

## Project Workflow

### 1. Data Fetching
- The Lambda function connects to the MySQL database and retrieves the dataset.
- Example:
```python
import pymysql

def fetch_data():
    connection = pymysql.connect(
        host='MYSQL_HOST',
        user='MYSQL_USER',
        password='MYSQL_PASSWORD',
        database='MYSQL_DATABASE'
    )
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM dataset_table")
    data = cursor.fetchall()
    connection.close()
    return data
```

### 2. Data Preprocessing
- The data is cleaned and tokenized for embedding generation.

### 3. Embedding Generation
- The OpenAI API is used to generate embeddings for the dataset:
```python
import openai

def generate_embeddings(text):
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=text
    )
    return response['data'][0]['embedding']
```

### 4. Storing in Pinecone
- Pinecone is used to store the generated embeddings:
```python
import pinecone

pinecone.init(api_key="PINECONE_API_KEY", environment="PINECONE_ENVIRONMENT")
index = pinecone.Index("your-index-name")

def store_embeddings(embedding, metadata):
    index.upsert([(metadata['id'], embedding, metadata)])
```

### 5. Querying and Answer Generation
- User queries are embedded and matched with stored data in Pinecone.
- The retrieved context is combined with the query to generate an answer using RAG.

Example:
```python
def generate_answer(query):
    query_embedding = generate_embeddings(query)
    results = index.query(query_embedding, top_k=5, include_metadata=True)
    context = " ".join([res['metadata']['text'] for res in results['matches']])

    prompt = f"Answer the question based on the context: {context}\n\nQuestion: {query}\nAnswer:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response['choices'][0]['text'].strip()
```

---

## Usage
1. Deploy the Lambda function.
2. Call the Lambda function with user queries through an API Gateway or any other trigger.
3. The chatbot will process the query, retrieve relevant context, and generate a response.

---

## Example Request and Response
**Request**:
```json
{
  "query": "What is the capital of France?"
}
```

**Response**:
```json
{
  "answer": "The capital of France is Paris."
}
```

---

## Contributing
Contributions are welcome! Please create a pull request or open an issue for suggestions and improvements.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgments
- [AWS Lambda](https://aws.amazon.com/lambda/)
- [OpenAI](https://openai.com/)
- [Pinecone](https://www.pinecone.io/)
- [MySQL](https://www.mysql.com/)