import streamlit as st
from matplotlib import pyplot as plt
import os
import re
import tempfile
from pathlib import Path
from llava_model import instantiate_model, forward_pass, get_processor, vit_attn_folder, generated_folder, forward_pass_one_step, attention_rollout, get_important_tokens, vit_attn_qkv_folder
import torch
import torch.nn.functional as F
from PIL import Image
import time
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import gc

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

# Memory tracker sidebar
memory_tracker_container = st.sidebar.empty()
with memory_tracker_container.container():
    st.sidebar.metric(label="cuda.memory_allocated", value=torch.cuda.memory_allocated(0)/1024/1024/1024)
    st.sidebar.metric(label="cuda.memory_reserved", value=torch.cuda.memory_reserved(0)/1024/1024/1024)
    st.sidebar.metric(label="cuda.max_memory_reserved", value=torch.cuda.max_memory_reserved(0)/1024/1024/1024)

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
            # Instantiate the model 
            model, processor, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id = instantiate_model()
            st.success("Model instantiated")
            # Run a forward pass
            output = forward_pass(model, processor, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id, tmp_file_path, prompt)
            st.success("Inference complete")
        
        # # Handle file upload if one exists in session state
        # if st.session_state.uploaded_file:

        #     st.session_state.uploader_key += 1  # Increment the key to force a reset of the file uploader
        #     st.session_state.uploaded_file = None  # Clear the uploaded file from session state

        # Save the output sequences and first forward pass LM attentions since its already generated anyways
    

        # Add assistant's response to conversation history and display it
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": processor.decode(output.sequences[0], skip_special_tokens=False)})
        with st.chat_message("assistant"):
            st.markdown(processor.decode(output.sequences[0], skip_special_tokens=False))

        # Delete the model variables to free up VRAM
        del model, processor, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id, output
        gc.collect()

        # Rerun the app to re-render the sidebar after updating the session state
        st.rerun()

with tab2:
    # Nested tabs
    tab5, tab6, tab7 = st.tabs(["ViT Attention", "ViT Q/K Vectors", "Attenion Rollout"])

    # ViT attention tab
    with tab5:
        # Band aid fix to prevent that stupid error
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        vit_layer = [i for i in range(24)]
        selected_vit_layer = st.selectbox(
            "ViT Attention Layer:",
            vit_layer,
            label_visibility="collapsed",
            key="select_box_vit_attention_tab"
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

    # Q/K Vectors tab
    with tab6:
        vit_layer = [i for i in range(24)]
        selected_vit_layer_qkv = st.selectbox(
            "ViT Attention Layer:",
            vit_layer,
            label_visibility="collapsed",
            key="select_box_qkv_tab"
        )
        # Load the qkv files 
        folder = Path(vit_attn_qkv_folder)
        files = [f for f in folder.iterdir() if f.is_file()]

        if len(files) == 0 or tmp_file_path is None:
            st.write("Upload an image and/or run the generation at least once to save the qkv vectors")
        else:
            # Get the q,k,v vectors according to selected layer
            q_vector, k_vector, v_vector = None, None, None
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            for f in files:
                    if f"layer_{str(selected_vit_layer_qkv)}_q" in str(f):
                        v = torch.load(os.path.join(vit_attn_qkv_folder, f))
                        v_pca = PCA(n_components=50, random_state=42).fit_transform(v[0].cpu().numpy())
                        q_vector = tsne.fit_transform(v_pca)

                    if f"layer_{str(selected_vit_layer_qkv)}_k" in str(f):
                        v = torch.load(os.path.join(vit_attn_qkv_folder, f))
                        v_pca = PCA(n_components=50, random_state=42).fit_transform(v[0].cpu().numpy())
                        k_vector = tsne.fit_transform(v_pca)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=q_vector[:,0],
                y=q_vector[:,1],
                mode='markers',
                marker=dict(
                    size=10,  # Adjust size as needed
                    color="green",  
                    line=dict(width=0.5, color='white')  # Optional border
                ),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=k_vector[:,0],
                y=k_vector[:,1],
                mode='markers',
                marker=dict(
                    size=10,  # Adjust size as needed
                    color="red",  
                    line=dict(width=0.5, color='white')  # Optional border
                ),
                showlegend=False
            ))
            
            st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="chart")



    # Attention rollout tab
    with tab7:
        # Load the generated text
        folder = Path(vit_attn_folder)
        files = [f for f in folder.iterdir() if f.is_file()]

        if len(files) == 0:
            st.write("Run a forward pass in Generation tab first")
        else:
            output_sequences = torch.load(os.path.join(generated_folder, "generated.pt"))
            processor = get_processor()

            generated_tokens_raw = processor.decode(output_sequences[0], skip_special_tokens=False)
            pattern = "<image>"
            count = len(re.findall(pattern, generated_tokens_raw))
            # Build the string to display
            generated_tokens_raw_after_image_tokens = generated_tokens_raw[7*count+10:]
            split = generated_tokens_raw_after_image_tokens.split("ASSISTANT:")
            user_prompt = f"<s>USER:<{count} image token(s)>{split[0].strip()} ASSISTANT:"
            st.title("User Prompt:")
            st.markdown(f"```\n{user_prompt}\n```")
            st.title("Output:")
            st.markdown(f"```\n{split[1].strip()}\n```")

            decoded_tokens = [processor.decode([t]) for t in output_sequences[0].tolist()]
            with st.expander("Full output"):
                st.write(decoded_tokens)

            st.title("Attention")
            file_path = os.path.join(generated_folder, "num.txt")
            with open(file_path, "r") as f:
                num_forward_pass = int(f.read())

            st.warning("Generating the rollout for a forward pass will instantiate the model and run a forward pass. Only generate 1 rollout at a time.")
            # List of expanders
            expanders = [i for i in range(-num_forward_pass, 0, 1)]

            # Initialize session state for lazy loading
            for exp in expanders:
                if f"show_{exp}" not in st.session_state:
                    st.session_state[f"show_{exp}"] = False
                if f"show_already_generated{exp}" not in st.session_state:
                    st.session_state[f"show_already_generated{exp}"] = False

            # Create each expander
            for exp in expanders:
                with st.expander(f"{decoded_tokens[exp]}"):
                    # Button inside each expander to trigger lazy load
                    if st.button(f"Load attention rollout", key=f"btn_{exp}"):
                        st.session_state[f"show_{exp}"] = True

                    # Show content only if triggered and content has not been generated 
                    if st.session_state[f"show_{exp}"] and not st.session_state[f"show_already_generated{exp}"]:
                        with st.spinner(f"Running inference for token: {decoded_tokens[exp]}"):
                            assistant_prompt = None
                            if exp != -num_forward_pass:
                                assistant_prompt = processor.decode(output_sequences[0][-num_forward_pass:exp], skip_special_tokens=False)
                            
                            model, processor_m, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id = instantiate_model()
                            st.success("Model instantiated")   
                            st.markdown(f"```\nAssistant prompt: {assistant_prompt}\n```")

                            file_path = os.path.join(generated_folder, "original_prompt.txt")
                            with open(file_path, "r") as f:
                                original_prompt = f.read()

                            output = forward_pass_one_step(model, processor_m, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id, tmp_file_path, original_prompt, assistant_prompt=assistant_prompt)

                        with st.expander("Attention Rollout"):
                            # Decode the next token
                            # Use -2 instead of -1 as the model appends a white space token to the end of the assistant
                            # prompt. Using -1 results in the wrong predicted token.
                            topk = torch.topk(output.logits[:, -2], k=1, dim=-1)
                            for ids in topk.indices:
                                st.markdown(f"```\nNext token is: {processor_m.tokenizer.decode(ids)}\n```")

                            # Run attention rollout on the enc_attn_weights
                            rollout = attention_rollout(model.enc_attn_weights[0:32])
                            raw_image = Image.open(tmp_file_path).convert("RGB")
                            heat_map_rollout, impt_text_tokens_index = get_important_tokens(rollout[-1], raw_image)
                            impt_text_tokens = [decoded_tokens[i] for i in impt_text_tokens_index]

                            fig, ax = plt.subplots()
                            ax.imshow(raw_image)
                            ax.imshow(heat_map_rollout, cmap='jet', alpha=0.75) 
                            ax.axis("off")

                            # Display in Streamlit
                            st.title("Important image tokens")
                            st.pyplot(fig)
                            st.title("Important text tokens")
                            st.write(impt_text_tokens)

                        with st.expander("Individual attention block"):
                            display_dict = {}
                            for i in range(len(output.hidden_states)):
                                display_dict.update({f"Attention layer {i}":{}})
                                logits_from_hidden_state = model.lm_head(output.hidden_states[i])
                                probs = probs = torch.softmax(logits_from_hidden_state, dim=-1)  
                                topk = torch.topk(probs[:, -2], k=10, dim=-1)
                                for ids, value in zip(topk.indices[0], topk.values[0]):
                                    display_dict[f"Attention layer {i}"].update({processor_m.tokenizer.decode(ids):value.item()})
                            st.write(display_dict)

                        del model, processor_m, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id, output
                        gc.collect()

                        # Set already_generated flag 
                        st.session_state[f"show_already_generated{exp}"] = True
    
with tab3:
    st.header("Interactive Maps")

with tab4:
    st.write("Misc")
