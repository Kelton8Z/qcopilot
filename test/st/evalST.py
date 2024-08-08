import streamlit as st

# Initialize session state if not already done
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'voted' not in st.session_state:
    st.session_state.voted = False

# Function to add a message
def add_message(role, content):
    st.session_state.messages.append({
        'role': role,
        'content': content
    })

# Function to handle voting
def vote(item):
    st.session_state.vote_item = item
    st.session_state.voted = True
    reason = st.text_input("Optional: Provide a reason for your vote")
    if st.button("Submit Vote"):
        add_message('system', f'User voted: {item}. Reason: {reason}')
        st.session_state.voted = False
        st.experimental_rerun()

# Input box for new messages (disabled if user hasn't voted yet)
prompt = st.text_input("Your question", disabled=not st.session_state.voted)
if prompt and st.session_state.voted:
    add_message("user", prompt)
    st.session_state.voted = False

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Simulate a chat engine response (replace with actual chat engine call)
            response = "This is a simulated response."
            st.write(response)
            add_message("assistant", response)
    st.session_state.voted = False
else:
    if not st.session_state.voted:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍"):
                vote("👍")
        with col2:
            if st.button("👎"):
                vote("👎")
