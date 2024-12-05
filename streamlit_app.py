import streamlit as st
from responses import response_generator
from recommender import load_data, select_random
import re
import os

st.title("Test Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize course data and embeddings (including intent embeddings)
if "data" not in st.session_state:
    st.session_state.data = load_data()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        #st.session_state.messages[-1]['state']

#with st.chat_message("supervisor"): # parameters "user" or "assistant" (for preset styling & avatar); otherwise can name it how I want
if len(st.session_state.messages) > 0:
    st.write(f"{st.session_state.messages[-1]['content']}, {st.session_state.messages[-1]['state']}")

# Accept user input
if prompt := st.chat_input("Start typing ..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    if len(st.session_state.messages) > 0:
        state = st.session_state.messages[-1]["state"]
    else:
        state = "default"

    if "nonsense" in prompt:
        state = "nonsense"
    else:
        state = "default"
    # Add user message to chat history
    st.write(f"Appending: {prompt} ({state})")
    st.write(select_random())
    st.session_state.messages.append({"role": "user", "content": prompt, "state": state})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(st.session_state.messages[-1]["state"]))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response, "state": state})