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
# df = pd.read_csv('harry_potter_topics.csv')

# Initialize the embeddings model
embedding_model = OpenAIEmbeddings()


# file_path = (
#     "harry_potter_topics.csv"
# )

# loader = CSVLoader(file_path=file_path)
# data = loader.load()

# for record in data[:2]:
#     print(record)

# Combine the text data into a single string
# text_data = ""
# for index, row in df.iterrows():
#     text_data += f"{row['text']}\n\n"

# Write the text data to a new text file
# output_file_path = "harry_potter_texts.txt"
# with open(output_file_path, "w") as file:
#     file.write(text_data)

# print(f"Text data has been written to {output_file_path}")
print("Start to open txt file")

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load example document
# with open("harry_potter_texts.txt") as f:
# Load book1 document

with open("01 Harry Potter and the Sorcerers Stone.txt") as f:

    harry_potter_texts = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # separators=[
    #     "\n\n",
    #     "\n",
    #     " ",
    #     ".",
    #     ",",
    #     "\u200b",  # Zero-width space
    #     "\uff0c",  # Fullwidth comma
    #     "\u3001",  # Ideographic comma
    #     "\uff0e",  # Fullwidth full stop
    #     "\u3002",  # Ideographic full stop
    #     "",
    # ],
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
documents = text_splitter.create_documents([harry_potter_texts])
# print(documents[0])
# print(documents[1])
# print(documents)

text_splitter.split_text(harry_potter_texts)[:3]

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

vectorstore_from_docs = PineconeVectorStore.from_documents(
        documents,
        index_name=index_name,
        embedding=embeddings
    )

# 可以添加多个txt file
# loader = TextLoader("../../modules/inaugural_address.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# vectorstore.add_documents(docs)

query = "Who is harry potter?"
vectorstore.similarity_search(query)

# Create vector store
docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)
time.sleep(1)

index = pc.Index(index_name)


for ids in index.list(namespace=namespace):
    query = index.query(
        id=ids[0],
        namespace=namespace,
        top_k=5,
        include_values=True,
        include_metadata=True
    )
    # print("first version query print", query)

# Using the fine-tuned model
from openai import OpenAI
client = OpenAI()

# completion = client.chat.completions.create(
#     model="ft:gpt-3.5-turbo-0125:personal::9wK1cery", 
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant knowledgeable about Harry Potter, particularly the first book, 'Harry Potter and the Sorcerer's Stone'."},
#         {"role": "user", "content": "Did Sirius escape Azkaban?"}
#     ]
# )
# print(completion.choices[0].message)

def get_answer(query):
    retriever = docsearch.as_retriever(search_kwargs={'k': 3})  # The retriever. K means Amount of documents to return (Default: 4)
    fine_tuned_model = ChatOpenAI(
    # openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model_name='gpt-4o-mini', #adjust to a new model
    temperature=0.2, # change the temperature 
    # model_name='ft:gpt-3.5-turbo-0125:personal::9wK1cery',
)

    system_prompt = (
    "You are a character from the Harry Potter universe base on the dataset. The user will choose between Harry, Ron, or Hermione, and you must respond as the chosen character. "
    "Adopt their tone, personality, and style of speech. For Harry, be courageous and determined, often reflecting on friendship and loyalty. "
    "For Ron, be humorous, a bit self-deprecating, and loyal, often referencing your family and love for food. "
    "For Hermione, be intelligent, logical, and thorough, often referencing books and knowledge. "
    "Answer using the context retrieved from the dataset"
    "If the user's query is out of the scope of the dataset(Harry Potter and the Sorcerers Stone), or you do not know the answer, politely say that you do not know."
    "If the context retrieved does not fully answer the question, try to provide the best answer based on the retrieved content. "
    "If you still cannot answer the question, politely say that you do not know."
    # "Do not speculate or provide information not included in the dataset"

    "\n\n"
    "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )

    
    question_answer_chain = create_stuff_documents_chain(fine_tuned_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    retriver2_result = rag_chain.invoke({"input": query})

    return retriver2_result




def retrieve_context_of_v2(query):
    print("=========chatbot retrieval context and generated answer=========")
    response = get_answer(query)
    print("Query:",query)
    print("\n\n")
    print("Generated Answer:",f"{response['answer']}")
    print("\n\n")
    print("Retrieve Contexts:",f"{response['context']}")
    print("\n\n")
    return response['context']



# Example queries
query1 = "Did Sirius escape Azkaban?"
# query2 = "where do you live before school?"
query2 = "Tell me about the end of Professor Snape"

query3 = "who sent you first birthday cake?"


# retrieve_contexts(query1,top_k=2)
retrieve_context_of_v2(query1)
retrieve_context_of_v2(query2)
retrieve_context_of_v2(query3)

