import os
import time
import pandas as pd
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader


from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI


# Set your API keys
os.environ['PINECONE_API_KEY'] = ''
os.environ['OPENAI_API_KEY'] = ''

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Define embeddings
model_name = 'text-embedding-3-small'
embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)

# Define index and namespace
index_name = "harry-potter-chatbot1"
namespace = "harryvector1"

# Setup Pinecone index
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Load the CSV file
df = pd.read_csv('harry_potter_topics.csv')

# Initialize the embeddings model
embedding_model = OpenAIEmbeddings()


file_path = (
    "harry_potter_topics.csv"
)

loader = CSVLoader(file_path=file_path)
data = loader.load()

for record in data[:2]:
    print(record)