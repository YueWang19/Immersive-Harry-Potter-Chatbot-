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

# # Start the fine-tuning process
# fine_tune_response = openai.FineTune.create(
#     training_file=file_id,
#     model="gpt-4o-mini",  # Replace with the base model you want to fine-tune
#     n_epochs=4  # Number of training epochs
# )
# fine_tune_id = fine_tune_response['id']
# print(f"Fine-tuning started with ID: {fine_tune_id}")

# # Optionally, monitor the fine-tuning process
# status = openai.FineTune.retrieve(id=fine_tune_id)
# print(f"Fine-tuning status: {status['status']}")
