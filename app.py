import streamlit as st
from matplotlib import pyplot as plt
import os
import tempfile
from pathlib import Path
from llava_model import forward_pass, clear_memory, vit_attn_folder
import torch
import torch.nn.functional as F
from PIL import Image

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
            clear_memory()
            processor, output = forward_pass(tmp_file_path, prompt)
            st.success("Inference complete")
        
        # # Handle file upload if one exists in session state
        # if st.session_state.uploaded_file:

        #     st.session_state.uploader_key += 1  # Increment the key to force a reset of the file uploader
        #     st.session_state.uploaded_file = None  # Clear the uploaded file from session state


        # Add assistant's response to conversation history and display it
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": processor.decode(output.sequences[0], skip_special_tokens=True)})
        with st.chat_message("assistant"):
            st.markdown(processor.decode(output.sequences[0], skip_special_tokens=True))

        # Rerun the app to re-render the sidebar after updating the session state
        st.rerun()

with tab2:
    clear_memory()
    # Band aid fix to prevent that stupid error
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    st.write("ViT Attention")

    vit_layer = [i for i in range(24)]
    selected_vit_layer = st.selectbox(
        "ViT Attention Layer:",
        vit_layer,
        label_visibility="collapsed"
    )
    folder = Path(vit_attn_folder)
    vit_attn_files = sorted(
        [f.name for f in folder.iterdir() if f.is_file()],
        key=lambda x: int(Path(x).stem)  # convert "3.pt" -> 3
    )

    if len(vit_attn_files) == 0:
        st.write("No ViT attn weights, run the generation first")
    else:
        if tmp_file_path is None:
            st.write("Upload image for display")
        else:
            attn = torch.load(os.path.join(vit_attn_folder, vit_attn_files[selected_vit_layer]))
            attn_avg = attn[0].mean(dim=0)
            cls_attn = attn_avg[0, 1:]  # exclude CLS itself
            H, W = 24, 24  # 336 / 14
            heatmap = cls_attn.reshape(H, W).detach().cpu().numpy()

            # Assuming heatmap shape [H, W]
            raw_image = Image.open(tmp_file_path).convert("RGB")
            heatmap_tensor = torch.tensor(heatmap[None, None], dtype=torch.float32)
            heatmap_full = F.interpolate(heatmap_tensor, size=(raw_image.size[1], raw_image.size[0]), mode='bilinear')[0,0].numpy()

            fig, ax = plt.subplots()
            ax.imshow(raw_image)
            ax.imshow(heatmap_full, cmap='jet', alpha=0.75) 
            ax.axis("off")

            # Display in Streamlit
            st.pyplot(fig)
    
with tab3:
    st.header("Interactive Maps")

with tab4:
    st.write("Misc")
