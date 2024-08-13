import streamlit as st
from harrychatbotv1 import get_answer

# Streamlit app starts here
st.title("Harry Potter Chatbot")

# Welcome message
st.write(
    "Welcome to the Harry Potter Chatbot! You can chat with one of the three characters: Harry, Ron, or Hermione. "
    "Please click the button corresponding to the character you'd like to talk to."
)

# Initialize session state for selected character and message history
if 'selected_character' not in st.session_state:
    st.session_state.selected_character = None

if 'message_history' not in st.session_state:
    st.session_state.message_history = []

# Buttons to select character
st.write("### Choose a character:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Harry"):
        st.session_state.selected_character = "Harry"
with col2:
    if st.button("Ron"):
        st.session_state.selected_character = "Ron"
with col3:
    if st.button("Hermione"):
        st.session_state.selected_character = "Hermione"

# Show a message after character selection
if st.session_state.selected_character:
    st.write(f"You are now talking to {st.session_state.selected_character}.")

    # Text input for user's query
    user_input = st.text_input(f"Ask a question to {st.session_state.selected_character}:")

    # Button to submit query
    if st.button("Ask"):
        if user_input:
            # Add user's question to the chat history
            user_message = f"You: {user_input}"
            st.session_state.message_history.append(user_message)
            
            # Modify the query to include the selected character
            query_with_character = f"{st.session_state.selected_character}: {user_input}"
            
            # Get the chatbot's response
            response = get_answer(query_with_character)['answer']
            
            # Add the chatbot's response to the chat history
            bot_message = f"{st.session_state.selected_character}: {response}"
            st.session_state.message_history.append(bot_message)
            
            # Use st.query_params to force a page update
            # st.experimental_set_query_params(updated="true")
            st.query_params['updated']="true"

        else:
            st.write("Please enter a question.")
    
    # Display the chat history in reverse order
    if st.session_state.message_history:
        st.write("### Chat History")
        for message in reversed(st.session_state.message_history):
            st.write(message)
else:
    st.write("Please select a character to start the conversation.")
