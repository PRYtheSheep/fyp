import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import tempfile
import time
from llava_model import forward_pass
import asyncio

# Set page configuration
st.set_page_config(
    page_title="FYP Visual analytics tool",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0  # counter for unique widget IDs

# Sidebar container for uploader only
uploader_container = st.sidebar.empty()

def render_uploader():
    with uploader_container.container():
        uploaded = st.file_uploader(
            "Upload an Image, Audio, or Video",
            type=["jpg", "jpeg", "png", "mp3", "wav", "mp4", "mov", "avi"],
            key=f"uploader_{st.session_state.uploader_key}",
        )
        
        tmp_file_path = None
        
        if uploaded:
            # New file uploaded
            st.session_state.uploaded_file = uploaded
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp_file:
                tmp_file.write(uploaded.read())
                tmp_file_path = tmp_file.name
        else:
            # No file (either never uploaded or X was clicked)
            st.session_state.uploaded_file = None
        
        # Show preview only if we currently have a file
        if st.session_state.get('uploaded_file'):
            file_type = st.session_state.uploaded_file.type
            if "image" in file_type:
                st.image(st.session_state.uploaded_file, caption="Selected Image")
            elif "audio" in file_type:
                st.audio(st.session_state.uploaded_file)
            elif "video" in file_type:
                st.video(st.session_state.uploaded_file)
        
        return uploaded, tmp_file_path

uploaded_file,tmp_file_path = render_uploader()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Generation", "Attention", "Hidden Representations", "Misc."])

# Tab 1: Data Analysis
with tab1:
    # Create dropdown from DataFrame column
    selected_model = st.selectbox(
        "Model:",
        [
            "LLaVa-1.5-7b"
        ],
        help="Choose a VLM"
    )

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                st.markdown(msg["content"])

    # Text input box
    if prompt := st.chat_input("Type your message..."):
        
        # Add user prompt to history
        st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})

        # return results
        with st.spinner("Running inference..."):
            processor, output = asyncio.run(forward_pass(tmp_file_path, prompt))
            st.success("Inference complete")
        
        # Handle file upload if one exists in session state
        if st.session_state.uploaded_file:

            st.session_state.uploader_key += 1  # Increment the key to force a reset of the file uploader
            st.session_state.uploaded_file = None  # Clear the uploaded file from session state


        # Add assistant's response to conversation history and display it
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": processor.decode(output.sequences[0], skip_special_tokens=True)})
        with st.chat_message("assistant"):
            st.markdown(processor.decode(output.sequences[0], skip_special_tokens=True))

        # Rerun the app to re-render the sidebar after updating the session state
        st.rerun()

# Tab 2: Form Input
with tab2:
    st.write("Attention")

# Tab 3: Maps
with tab3:
    st.header("Interactive Maps")

# Tab 4: Settings
with tab4:
    st.write("Misc")
