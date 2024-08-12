import streamlit as st
from harrychatbotv1 import get_answer

# Streamlit app starts here
st.title("Harry Potter Chatbot")

# Welcome message
st.write(
    "Welcome to the Harry Potter Chatbot! You can chat with one of the three characters: Harry, Ron, or Hermione. "
    "Please click the button corresponding to the character you'd like to talk to."
)

# Initialize session state for selected character
if 'selected_character' not in st.session_state:
    st.session_state.selected_character = None

# Buttons to select character
if st.button("Harry"):
    st.session_state.selected_character = "Harry"
elif st.button("Ron"):
    st.session_state.selected_character = "Ron"
elif st.button("Hermione"):
    st.session_state.selected_character = "Hermione"

# Show a message after character selection
if st.session_state.selected_character:
    st.write(f"You are now talking to {st.session_state.selected_character}.")
    
    # Text input for user's query
    user_input = st.text_input(f"Ask a question to {st.session_state.selected_character}:")

    # Button to submit query
    if st.button("Submit"):
        if user_input:
            # Modify the query to include the selected character
            query_with_character = f"{st.session_state.selected_character}: {user_input}"
            # Get the chatbot's response
            response = get_answer(query_with_character)['answer']
            # Display the response in Streamlit
            st.write("Chatbot Response:")
            st.write(response)
        else:
            st.write("Please enter a question.")
else:
    st.write("Please select a character to start the conversation.")
