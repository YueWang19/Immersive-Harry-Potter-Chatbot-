# Immersive-Harry-Potter-Chatbot-

# Harry Potter Themed Chatbot: A Generative AI Application

This project showcases a Harry Potter-themed chatbot that allows users to engage in immersive conversations with characters like Harry, Ron, and Hermione. By leveraging cutting-edge AI technologies, including Generative AI, Retrieval-Augmented Generation (RAG), LangChain, and fine-tuning models, this project brings the world of Harry Potter to life through interactive dialogues.

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Setup](#setup)
- [Usage](#usage)
- [Fine-Tuning Process](#fine-tuning-process)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Harry Potter Themed Chatbot is an innovative application that combines various AI technologies to create a dynamic conversational agent capable of impersonating characters from the Harry Potter universe. The project is designed to demonstrate the potential of AI in entertainment and education by providing fans with a unique and engaging way to interact with their favorite characters.

You can watch the introduction by this link:

https://www.youtube.com/watch?v=yyXmHPxSgc0

## Key Features

- **Character-Based Interaction**: Users can select to talk to Harry, Ron, or Hermione, and the chatbot responds in a manner consistent with the chosen character's personality and tone.
- **Contextual Responses Using RAG**: The chatbot uses Retrieval-Augmented Generation to ensure that responses are grounded in the text of "Harry Potter and the Sorcerer's Stone."
- **Fine-Tuned Model for Enhanced Performance**: A custom fine-tuned version of GPT-3.5-turbo was used to improve the character-specific responses.
- **Dynamic Interface**: A user-friendly interface built with Streamlit, maintaining chat history and allowing for continuous conversation.

## Technologies Used

- **Generative AI**: GPT-3.5-turbo and GPT-4o-mini models from OpenAI.
- **Retrieval-Augmented Generation (RAG)**: Combines retrieval of relevant documents with generation of responses.
- **LangChain**: Framework to manage the flow of information between components.
- **Pinecone**: Vector store used to efficiently store and retrieve text data.
- **Streamlit**: Used to build the frontend interface of the chatbot.
- **Python**: The core language used for development.

## Setup

To set up this project locally, follow these steps:

### Prerequisites

- Python 3.12 or later
- An OpenAI API key
- A Pinecone API key

### Installation

1. **Clone the Repository:**

```bash
git clone https://github.com/YueWang19/Immersive-Harry-Potter-Chatbot-.git
cd Immersive-Harry-Potter-Chatbot-
```

2. **Install Dependencies:**

   Use pip to install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables:**

Create a .env file in the root directory and add your API keys:

```bash
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
```

4. **Prepare Dataset:**
   Ensure you have the 01 Harry Potter and the Sorcerers Stone.txt file in the correct directory for processing and fine-tuning.

5. **Fine-Tuning (Optional):**
   If you want to fine-tune the model on new data, follow the instructions in the Fine-Tuning Process section below.

### Usage

1. **To run the chatbot application:**

Start the Backend:
Ensure that the `harrychatbotv1.py` script is properly configured and start the backend by running:

```bash
python harrychatbotv1.py
```

2. **Launch the Streamlit Frontend:**
   In a separate terminal, start the Streamlit app by running:

```bash
streamlit run app.py
```

3. **Interact with the Chatbot:**
   Open the URL provided by Streamlit in your web browser, select a character, and start asking questions!

### Fine-Tuning Process

The fine-tuning process involves customizing the GPT-3.5-turbo model with a specific dataset designed to improve the chatbot's ability to respond in character-specific ways.

Steps:

1. **Prepare the Dataset:**
   Create a .jsonl file with the required training data. An example of the dataset structure can be found in the harry_potter_finetune_data.jsonl file.

2. **Upload the Dataset:**
   Use the OpenAI API to upload the dataset for fine-tuning.

3. **Fine-Tune the Model:**
   Start the fine-tuning job using the OpenAI API, specifying the training dataset and model type.

4. **Monitor the Fine-Tuning Process:**
   Track the progress, including training loss and steps, through the OpenAI dashboard.

5. **Deploy the Fine-Tuned Model:**
   Once fine-tuning is complete, update the harrychatbotv1.py to use the newly fine-tuned model.

### Future Improvements

Potential Enhancements:

1. Expand Dataset: Include more content from other books in the Harry Potter series to improve the chatbot's response accuracy.

2. Hybrid Model Approach: Combine the fine-tuned model with a more generalized model like GPT-4o-mini to create a robust chatbot that can handle a wider range of queries.

3. Multimodal Interactions: Integrate visual and auditory elements to further enhance the user experience.

4. Refinement of Fine-Tuning Process: Experiment with different fine-tuning strategies and dataset configurations to improve the chatbot's ability to generalize across different types of queries.

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### License

This project is licensed under the MIT License

### Contact me

yueyuelala19@gmail.com

https://www.linkedin.com/in/yuewang19/
