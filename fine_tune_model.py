import os
from openai import OpenAI

# Set your API key
# openai.api_key = 'your-openai-api-key'
os.environ['OPENAI_API_KEY'] = ''

client = OpenAI()
# Upload the dataset
response = client.files.create(
    file=open("harry_potter_finetune_data.jsonl","rb"),
    purpose='fine-tune'
)
# file_id = response['id']
# print(f"Uploaded file ID: {file_id}")
print("1. After upload file========",response)

#create a fine-turned model


# Start a fine-tuning job
jobresponse = client.fine_tuning.jobs.create(
    training_file=response.id,  # Replace 'file-abc123' with your actual file ID
    model="gpt-3.5-turbo"  # Model type to fine-tune
)
print("2. Start a fine-tuning job========",jobresponse)


# Retrieve the state of a fine-tune job
job_status = client.fine_tuning.jobs.retrieve(jobresponse.id)  # Replace 'ftjob-abc123' with your actual job ID
print("3. Job status print========",job_status)


