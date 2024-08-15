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
print(response)

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
