import streamlit as st
from matplotlib import pyplot as plt
import os
import re
import tempfile
from pathlib import Path
from llava_model import instantiate_model, forward_pass, get_processor, vit_attn_folder, generated_folder, forward_pass_one_step, attention_rollout, get_important_tokens, vit_attn_qkv_folder
import torch
from torch import Tensor
import torch.nn.functional as F
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import gc 
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAEConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)

from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float
import pandas as pd

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
if "gemma_output_no_steering" not in st.session_state:
    st.session_state.gemma_output_no_steering = None  
if "gemma_output_steering" not in st.session_state:
    st.session_state.gemma_output_steering = None  
if "gemma_residual_mid_no_steering" not in st.session_state:
    st.session_state.gemma_residual_mid_no_steering = None  
if "gemma_residual_post_no_steering" not in st.session_state:
    st.session_state.gemma_residual_post_no_steering = None  
if "gemma_residual_mid_steering" not in st.session_state:
    st.session_state.gemma_residual_mid_steering = None  
if "gemma_residual_post_steering" not in st.session_state:
    st.session_state.gemma_residual_post_steering = None   
if "gemma_sae_acts_no_steering" not in st.session_state:
    st.session_state.gemma_sae_acts_no_steering = None  
if "gemma_sae_acts_steering" not in st.session_state:
    st.session_state.gemma_sae_acts_steering = None  
if "gemma_sae_recons_no_steering" not in st.session_state:
    st.session_state.gemma_sae_recons_no_steering = None  
if "gemma_sae_recons_steering" not in st.session_state:
    st.session_state.gemma_sae_recons_steering = None  

# Memory tracker sidebar
memory_tracker_container = st.sidebar.empty()
with memory_tracker_container.container():
    st.sidebar.metric(label="cuda.memory_allocated", value=torch.cuda.memory_allocated(0)/1024/1024/1024)
    st.sidebar.metric(label="cuda.memory_reserved", value=torch.cuda.memory_reserved(0)/1024/1024/1024)
    st.sidebar.metric(label="cuda.max_memory_reserved", value=torch.cuda.max_memory_reserved(0)/1024/1024/1024)

# Sidebar container for uploader only
uploader_container = st.sidebar.empty()

def get_rank_data(token_input, gemma_model):
    """Helper to process ranks and return a DataFrame"""
    results = []
    # Using your existing logic to zip layer items
    zipped_layers = zip(
        st.session_state.gemma_residual_mid_no_steering.items(), 
        st.session_state.gemma_residual_post_no_steering.items()
    )
    
    for i, (r_mid_item, r_post_item) in enumerate(zipped_layers):
        tokens_mid = unembed_gemma(r_mid_item[1], gemma_model, 20)
        tokens_post = unembed_gemma(r_post_item[1], gemma_model, 20)
        
        # Calculate ranks (helper to keep things dry)
        r_mid = list(tokens_mid.keys()).index(token_input) if token_input in tokens_mid else -1
        r_post = list(tokens_post.keys()).index(token_input) if token_input in tokens_post else -1
        
        results.append({"Layer": i, "Mid Rank": r_mid, "Post Rank": r_post})
    
    return pd.DataFrame(results)

def color_ranks(val):
    """
    Returns CSS for background color based on rank.
    Rank 0: Darkest | Rank 20: Lightest | -1: None
    """
    if val == -1:
        return ""
    # Calculate opacity: Rank 0 -> 0.9, Rank 20 -> 0.1
    opacity = max(0.1, 1.0 - (val / 22)) 
    return f"background-color: rgba(255, 75, 75, {opacity}); color: {'white' if opacity > 0.5 else 'black'}"

def render_uploader():
    with uploader_container.container():
        uploaded = st.file_uploader(
            "Image only",
            type=["jpg", "jpeg", "png"],
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

@st.cache_resource
def instantiate_model_gemma():
    """
    Instantiates a Gemma model and returns the model and SAE. Only instantiate 1 model at a time due to GPU memory limits.
    Use del to delete the model and SAE before instantiating a new model
    """
    gemmascope_sae_release = "gemma-scope-2b-pt-mlp-canonical"
    gemmascope_all_sae_ids = [
        f"layer_{i}/width_16k/canonical" for i in range(0,26)
    ]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    gemma_2_2b = HookedSAETransformer.from_pretrained("gemma-2-2b", device=device)
    gemma_2_2b_all_sae = [
        SAE.from_pretrained(gemmascope_sae_release, i, device=str(device))[0] for i in gemmascope_all_sae_ids
    ]
    return gemma_2_2b, gemma_2_2b_all_sae

def run_gemma_with_hook_with_steering(gemma_2_2b, gemma_2_2b_all_sae, prompt, steering_args=None):
    """
    Calls a single forward pass with gemma model, hooks the model and runs with steering. Only steer 1 laten_idx at a time.
    Model and SAE hooks are reset before new hooks are applied.
    """
    # Reset hooks
    gemma_2_2b.reset_hooks()

    # Attach hooks to extract residuals, SAE activations and SAE reconstructions
    residual_mid_after_steering = {}
    def capture_residual_mid_hook(layer):
        def hook_fn(
            activation: Float[Tensor, "batch pos d_model"],
            hook: HookPoint,
        ) -> None:
            """Capture residual stream for specific layer."""
            print(f"extracting resid mid from layer {layer}")
            residual_mid_after_steering[layer] = activation.detach()
        return hook_fn

    residual_mid_hooks_after_steering = []
    for layer in range(26):
        residual_mid_hooks_after_steering.append((f"blocks.{layer}.hook_resid_mid", capture_residual_mid_hook(layer)))

    residual_post_after_steering = {}
    def capture_residual_post_hook(layer):
        def hook_fn(
            activation: Float[Tensor, "batch pos d_model"],
            hook: HookPoint,
        ) -> None:
            """Capture residual stream for specific layer."""
            print(f"extracting resid post from layer {layer}")
            residual_post_after_steering[layer] = activation.detach()
        return hook_fn

    residual_post_hooks_after_steering = []
    for layer in range(26):
        residual_post_hooks_after_steering.append((f"blocks.{layer}.hook_resid_post", capture_residual_post_hook(layer)))

    captured_sae_acts_post = {}
    def extract_sae_acts_post(layer):
        def hook_fn(activation, hook):
            print(f"extracting sae acts_post from layer {layer}")
            captured_sae_acts_post[layer] = activation.detach().cpu()
            return activation
        return hook_fn

    captured_sae_recons = {}
    def extract_sae_recons(layer):
        def hook_fn(activation, hook):
            print(f"extracting sae recons from layer {layer}")
            captured_sae_recons[layer] = activation.detach().cpu()
            return activation
        return hook_fn

    for layer, sae in enumerate(gemma_2_2b_all_sae):
        sae.reset_hooks()
    for layer, sae in enumerate(gemma_2_2b_all_sae):
        sae.add_hook("hook_sae_acts_post", extract_sae_acts_post(layer))
        sae.add_hook("hook_sae_recons", extract_sae_recons(layer))

    if steering_args is not None:
        # Add steering hooks and steer the model
        latent_idx = steering_args["latent_idx"]
        layer_idx = steering_args["layer_idx"]
        steering_coefficient = steering_args["steering_coefficient"]

        def sae_steering(latent_idx, layer_idx, steering_coefficient, sae):
            def hook_fn(
                activations: Float[Tensor, "batch pos d_in"],
                hook: HookPoint,
            ) -> Tensor:
                """
                Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
                sequence positions.
                """
                print(f"steering at layer {layer_idx}")
                activations[:,:,latent_idx] =  steering_coefficient
                return activations
            return hook_fn
        gemma_2_2b_all_sae[layer_idx].add_hook("hook_sae_acts_post", sae_steering(latent_idx,layer_idx,steering_coefficient, gemma_2_2b_all_sae[layer_idx]))

    output = gemma_2_2b.run_with_hooks_with_saes(
        prompt,
        fwd_hooks=[*residual_mid_hooks_after_steering, *residual_post_hooks_after_steering],
        saes=gemma_2_2b_all_sae,
        reset_saes_end=True,
        reset_hooks_end=True
    )
    gemma_2_2b.reset_hooks()
    return output, residual_mid_after_steering, residual_post_after_steering, captured_sae_acts_post, captured_sae_recons

def unembed_gemma(resid, gemma_2_2b, k):
    activations_norm_ln_final=gemma_2_2b.ln_final(resid)
    my_logits = gemma_2_2b.unembed(activations_norm_ln_final)
    my_logits_softcap = gemma_2_2b.cfg.output_logits_soft_cap * F.tanh(my_logits / gemma_2_2b.cfg.output_logits_soft_cap)
    probs = torch.softmax(my_logits_softcap[0, -1, :], dim=-1)
    top_probs, top_indices = torch.topk(probs, k=k)
    tokens = {}
    for i in range(k):
        token = gemma_2_2b.to_string(top_indices[i]).strip()
        tokens[token] = top_probs[i].item()
    return tokens

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Generation", "Attention", "Mechanistic Interpretability", "Misc."])

with tab1:
    # Create dropdown from DataFrame column
    selected_model = st.selectbox(
        "Model:",
        [
            "LLaVa-1.5-7b",
            "Gemma-2-2b"
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
            if selected_model == "LLaVa-1.5-7b":
                # Instantiate the model 
                model, processor, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id = instantiate_model()
                st.success("Model instantiated")
                # Run a forward pass
                output = forward_pass(model, processor, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id, tmp_file_path, prompt)
                st.success("Inference complete")

            elif selected_model == "Gemma-2-2b":
                gemma_2_2b, gemma_2_2b_all_sae =  instantiate_model_gemma()
                output, residual_mid, residual_post, captured_sae_acts_post, captured_sae_recons = run_gemma_with_hook_with_steering(gemma_2_2b, gemma_2_2b_all_sae, prompt, None)
                st.session_state.gemma_output_no_steering = output
                st.session_state.gemma_residual_mid_no_steering = residual_mid
                st.session_state.gemma_residual_post_no_steering = residual_post
                st.session_state.gemma_sae_acts_no_steering = captured_sae_acts_post
                st.session_state.gemma_sae_recons_no_steering = captured_sae_recons
        
        if selected_model == "LLaVa-1.5-7b":
            # Add assistant's response to conversation history and display it
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": processor.decode(output.sequences[0], skip_special_tokens=False)})
            with st.chat_message("assistant"):
                st.markdown(processor.decode(output.sequences[0], skip_special_tokens=False))

            # Delete the model variables to free up VRAM
            del model, processor, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id, output
            gc.collect()

            # Rerun the app to re-render the sidebar after updating the session state
            st.rerun()
        
        elif selected_model == "Gemma-2-2b":
            # Add assistant's response to conversation history and display it
            probs = torch.softmax(output[0, -1, :], dim=-1)
            top_probs, top_indices = torch.topk(probs, k=10)
            lines = {}
            for i in range(len(top_indices)):
                # Decode the token
                token_str = gemma_2_2b.tokenizer.decode(top_indices[i]).strip()
                
                # Make invisible characters visible for debugging
                display_token = repr(token_str)
                
                # Use f-string padding to keep percentages aligned
                lines[token_str] = top_probs[i].item()
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"```{lines}```"})
            with st.chat_message("assistant"):
                st.write(lines)
            st.rerun()

with tab2:
    # Nested tabs
    tab5, tab6, tab7 = st.tabs(["ViT Attention", "ViT Q/K Vectors", "LM Attenion"])

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
                        v = torch.load(os.path.join(vit_attn_qkv_folder, f))[:, 1:] 
                        v_pca = PCA(n_components=50, random_state=42).fit_transform(v[0].cpu().numpy())
                        q_vector = tsne.fit_transform(v_pca) # Shape 576,2

                    if f"layer_{str(selected_vit_layer_qkv)}_k" in str(f):
                        v = torch.load(os.path.join(vit_attn_qkv_folder, f))[:, 1:] 
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
            
            # Create 2 columns, one for plotly one for image
            col_graph, col_image = st.columns([1, 1.5])
            with col_graph:
                clicked_point = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="vit_qk_chart")
            with col_image:
                raw_image = Image.open(tmp_file_path).convert("RGB")
                fig, ax = plt.subplots()
                
                # Initialize three separate overlay masks
                q_mask = np.zeros(576)
                k_mask = np.zeros(576)
                
                if clicked_point and clicked_point.selection and clicked_point.selection.points:
                    for p in clicked_point.selection.points:
                        idx = p['point_index']
                        # curve_number 0 = Green (Queries), curve_number 1 = Red (Keys)
                        if p['curve_number'] == 0:
                            q_mask[idx] = 1
                        elif p['curve_number'] == 1:
                            k_mask[idx] = 1

                    # Logic for colors:
                    # 1. Intersection (Both selected) -> Blue (or any color you like)
                    # 2. Only Q -> Green
                    # 3. Only K -> Red
                    
                    # Create an RGB overlay (H, W, 3)
                    overlay = np.zeros((24, 24, 3))
                    
                    for i in range(576):
                        r, c = divmod(i, 24)
                        if q_mask[i] and k_mask[i]:
                            overlay[r, c] = [0, 0, 1] # Blue for both
                        elif q_mask[i]:
                            overlay[r, c] = [0, 1, 0] # Green for Q
                        elif k_mask[i]:
                            overlay[r, c] = [1, 0, 0] # Red for K

                    # Resize the overlay to match image size using Nearest Neighbor to keep blocks sharp
                    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8)).resize(raw_image.size, resample=Image.NEAREST)
                    
                    # Show image and the custom RGB overlay
                    ax.imshow(raw_image)
                    ax.imshow(overlay_img, alpha=0.5)
                    ax.axis('off')
                else:
                    ax.imshow(raw_image)
                    ax.axis('off')
                
                st.pyplot(fig)



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
                if f"expander_open_{exp}" not in st.session_state:
                        st.session_state[f"expander_open_{exp}"] = False
                with st.expander(f"{decoded_tokens[exp]}", expanded=st.session_state[f"expander_open_{exp}"]):

                    # Button inside each expander to trigger lazy load
                    if st.button(f"Load attention rollout", key=f"btn_{exp}"):
                        st.session_state[f"show_{exp}"] = True
                        st.session_state[f"expander_open_{exp}"] = True

                    # Show content only if triggered and content has not been generated 
                    if st.session_state[f"show_{exp}"]:
                        if not st.session_state[f"show_already_generated{exp}"]:
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
                                st.session_state["model"] = model
                                st.session_state["output"] = output
                                st.session_state["processor_m"] = processor_m

                        with st.expander("Attention Rollout"):
                            # Decode the next token
                            # Use -2 instead of -1 as the model appends a white space token to the end of the assistant
                            # prompt. Using -1 results in the wrong predicted token.
                            topk = torch.topk(st.session_state["output"].logits[:, -2], k=1, dim=-1)
                            for ids in topk.indices:
                                st.markdown(f"```\nNext token is: {st.session_state['processor_m'].tokenizer.decode(ids)}\n```")

                            # Run attention rollout on the enc_attn_weights
                            rollout = attention_rollout(st.session_state["model"].enc_attn_weights[0:32])
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
                            for i in range(len(st.session_state["output"].hidden_states)):
                                display_dict.update({f"Attention layer {i}":{}})
                                logits_from_hidden_state = st.session_state["model"].lm_head(st.session_state["output"].hidden_states[i])
                                probs = probs = torch.softmax(logits_from_hidden_state, dim=-1)  
                                topk = torch.topk(probs[:, -2], k=10, dim=-1)
                                for ids, value in zip(topk.indices[0], topk.values[0]):
                                    display_dict[f"Attention layer {i}"].update({st.session_state["processor_m"].tokenizer.decode(ids):value.item()})
                            st.write(display_dict)

                        with st.expander("Q/K Vectors"):
                            #############
                            st.write("Tokens up till this point:")
                            st.write(decoded_tokens[:exp])
                            #############
                            llm_layer = [i for i in range(32)]
                            selected_llm_layer_qkv = st.selectbox(
                                "LLM Attention Layer:",
                                llm_layer,
                                label_visibility="collapsed",
                                key="select_box_llm_qkv_tab"
                            )
                            llm_q_vectors = st.session_state["model"].llm_qkv_vectors[0][f"layer_{str(selected_llm_layer_qkv)}_q_proj"][0]
                            llm_k_vectors = st.session_state["model"].llm_qkv_vectors[0][f"layer_{str(selected_llm_layer_qkv)}_k_proj"][0]

                            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                            q_pca = PCA(n_components=50, random_state=42).fit_transform(llm_q_vectors.cpu().numpy())
                            q_vector = tsne.fit_transform(q_pca)
                            k_pca = PCA(n_components=50, random_state=42).fit_transform(llm_k_vectors.cpu().numpy())
                            k_vector = tsne.fit_transform(k_pca)

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=q_vector[:,0],
                                y=q_vector[:,1],
                                mode='markers+text',
                                name='q', # curve number 0
                                text=decoded_tokens[:exp],  # label per point
                                textposition="top center",
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
                                mode='markers+text',
                                name='k', # curve number 1
                                text=decoded_tokens[:exp],  # label per point
                                textposition="top center",
                                marker=dict(
                                    size=10,  # Adjust size as needed
                                    color="red",  
                                    line=dict(width=0.5, color='white')  # Optional border
                                ),
                                showlegend=False
                            ))
                            # --------------------------------------
                            def on_select():
                                sel = st.session_state["llm_qk_chart"]["selection"]
                                st.session_state.selected_tokens = [
                                    p["point_index"] for p in sel["points"]
                                ]
                            # --------------------------------------
                            st.plotly_chart(fig, use_container_width=True, on_select=on_select, key="llm_qk_chart")
                            if "selected_tokens" in st.session_state:
                                st.write(st.session_state.selected_tokens)

                        # del model, processor_m, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id, output
                        gc.collect()

                        # Set already_generated flag 
                        st.session_state[f"show_already_generated{exp}"] = True
    
with tab3:
    st.set_page_config(
        layout="wide"
    )

    nodes = [StreamlitFlowNode(id='in', pos=(100, 100), data={'content': 'Input Tokens'}, node_type='input', source_position='right', draggable=False),]

    for i in range(0,26):
        nodes.extend([
            StreamlitFlowNode(id=f'attn_{i}', pos=(250+200*i, 100), data={'content': f'ATN{i}'}, node_type='default', source_position='right', target_position='left', draggable=False),
            StreamlitFlowNode(id=f'mlp_{i}', pos=(315+200*i, 100), data={'content': f'MLP{i}'}, node_type='default', source_position='right', target_position='left', draggable=False),
            StreamlitFlowNode(id=f'rsd_{i}', pos=(391+200*i, 108), data={}, node_type='default', source_position='right', target_position='left', draggable=False, style={'width':40, 'height':40})
            ])
        
    nodes.extend([
        StreamlitFlowNode(id=f'unembed', pos=(250+200*26, 100), data={'content': f'Unembedding'}, node_type='default', source_position='right', target_position='left', draggable=False),
        StreamlitFlowNode(id=f'out', pos=(5450+120, 100), data={'content': f'Output Token'}, node_type='default', source_position='right', target_position='left', draggable=False),
    ])

    edges = [
        # StreamlitFlowEdge('in-connector_0', 'in', 'connector_0', animated=True),
        # StreamlitFlowEdge('connector_0-attn_0', 'connector_0', 'attn_0', animated=True),
        # StreamlitFlowEdge('attn_0-mlp_0', 'attn_0', 'mlp_0', animated=True),
    ]

    if 'click_interact_state' not in st.session_state:
        st.session_state.click_interact_state = StreamlitFlowState(nodes, edges)

    updated_state = streamlit_flow('ret_val_flow',
                    st.session_state.click_interact_state,
                    fit_view=True,
                    get_node_on_click=True,
                    get_edge_on_click=True)

    st.write(f"Clicked on: {updated_state.selected_id}")
    col1, col2 = st.columns([2, 1])
    with col1:
        with st.expander("Activations"):
            gemma_2_2b, gemma_2_2b_all_sae =  instantiate_model_gemma()
            st.markdown(f"```\nThis graph plots the tensor from the last row of the residual stream, taken after each layer.\n```")
            if st.session_state.gemma_residual_mid_no_steering is None:
                st.markdown("```Run a forward pass in the generations tab with the Gemma model first```")
            else:
                pca = PCA(n_components=2, random_state=42)
                # Use the unsteered residual output as the base, each residual has shape 1, seq_len, 2304
                refs = []
                for layer in st.session_state.gemma_residual_mid_no_steering.keys():
                    refs.append(st.session_state.gemma_residual_mid_no_steering[layer][0].cpu())
                    refs.append(st.session_state.gemma_residual_post_no_steering[layer][0].cpu())

                ref = np.concatenate(refs, axis=0)
                pca.fit(ref)

                # Plot each layer's last sequence activation
                fig = go.Figure() # Initialise the graph

                # Plot the base model traj
                traj_mid = np.array([
                    pca.transform(r[0].cpu())[-1]
                    for r in st.session_state.gemma_residual_mid_no_steering.values()
                ])

                traj_post = np.array([
                    pca.transform(r[0].cpu())[-1]
                    for r in st.session_state.gemma_residual_post_no_steering.values()
                ])
                for layer_idx, (mid, post) in enumerate(zip(traj_mid, traj_post)):
                    r_mid, r_post = st.session_state.gemma_residual_mid_no_steering[layer_idx], st.session_state.gemma_residual_post_no_steering[layer_idx]
                    token_mid, token_post = unembed_gemma(r_mid, gemma_2_2b, 20), unembed_gemma(r_post, gemma_2_2b, 20)
                    hovertext_mid, hovertext_post = f"Layer {layer_idx} resid_mid", f"Layer {layer_idx} resid_post"
                    for token, prob in token_mid.items():
                        hovertext_mid += f"<br>{token}: {prob:.2%}:"
                    for token, prob in token_post.items():
                        hovertext_post += f"<br>{token}: {prob:.2%}:"

                    fig.add_trace(go.Scatter( # add the traces for the residual points here
                        x=[mid[0], post[0]],
                        y=[mid[1], post[1]],
                        mode="lines+markers",
                        line=dict(color="blue", width=2),
                        marker=dict(color="blue", size=[6, 8]),
                        hovertext=[
                            f"{hovertext_mid}",
                            f"{hovertext_post}",
                            # Add in the output tokens for that layer here
                        ],
                        hoverinfo="text",
                        name=f"Layer {layer_idx} residual",
                        showlegend=True,
                    ))
                # Plot the steered model traj
                if st.session_state.gemma_residual_mid_steering is not None and st.session_state.gemma_residual_post_steering is not None:
                    traj_mid_steered = np.array([
                        pca.transform(r[0].cpu())[-1]
                        for r in st.session_state.gemma_residual_mid_steering.values()
                    ])

                    traj_post_steered = np.array([
                        pca.transform(r[0].cpu())[-1]
                        for r in st.session_state.gemma_residual_post_steering.values()
                    ])
                    for layer_idx, (mid, post) in enumerate(zip(traj_mid_steered, traj_post_steered)):
                        r_mid, r_post = st.session_state.gemma_residual_mid_steering[layer_idx], st.session_state.gemma_residual_post_steering[layer_idx]
                        token_mid, token_post = unembed_gemma(r_mid, gemma_2_2b, 20), unembed_gemma(r_post, gemma_2_2b, 20)
                        hovertext_mid, hovertext_post = f"Layer {layer_idx} resid_mid", f"Layer {layer_idx} resid_post"
                        for token, prob in token_mid.items():
                            hovertext_mid += f"<br>{token}: {prob:.2%}:"
                        for token, prob in token_post.items():
                            hovertext_post += f"<br>{token}: {prob:.2%}:"

                        fig.add_trace(go.Scatter( # add the traces for the residual points here
                            x=[mid[0], post[0]],
                            y=[mid[1], post[1]],
                            mode="lines+markers",
                            line=dict(color="green", width=2),
                            marker=dict(color="green", size=[6, 8]),
                            hovertext=[
                                f"{hovertext_mid}",
                                f"{hovertext_post}",
                                # Add in the output tokens for that layer here
                            ],
                            hoverinfo="text",
                            name=f"Layer {layer_idx} residual",
                            showlegend=True,
                        ))

                # Plot each layer's top 5 features directions
                _, gemma_2_2b_all_sae = instantiate_model_gemma()
                # For each layer's feature activation
                features = {} # Dict to hold all raw feature directions to calculate cosine similarity
                for layer_idx, sae_act in st.session_state.gemma_sae_acts_no_steering.items():
                    top_indices = np.argsort(np.abs(sae_act.numpy()[0, -1]))[-10:][::-1]
                    top_indices_strength = sae_act.numpy()[0, -1][top_indices]

                    x0, y0 = traj_mid[layer_idx, :]
                    xs, ys, hovertexts = [], [], []
                    features_layer = {}

                    for latent_idx, strength in zip(top_indices, top_indices_strength):
                        threshold = gemma_2_2b_all_sae[layer_idx].threshold[latent_idx]
                        if strength < threshold:
                            continue
                        feature_direction = (
                            (strength - threshold).detach()
                            * gemma_2_2b_all_sae[layer_idx].W_dec[latent_idx]
                                .detach()
                        )
                        features_layer[int(latent_idx)] = feature_direction
                        feature_2d = feature_direction.cpu().numpy() @ pca.components_.T
                        dx, dy = feature_2d
                        x1 = x0 + dx * 0.7
                        y1 = y0 + dy * 0.7
                        # Add line segment
                        xs.extend([x0, x1, None])
                        ys.extend([y0, y1, None])
                        # Get the resid mid and add the feature to it, then unembed it
                        r_mid = st.session_state.gemma_residual_mid_no_steering[layer_idx]
                        r_mid_plus_feature = r_mid + feature_direction
                        token_mid_plus_feature = unembed_gemma(r_mid_plus_feature, gemma_2_2b, 20)
                        hovertext = f"Layer {layer_idx}<br>Feature {latent_idx}<br>Strength {(strength - threshold).item():.3f}"
                        for token, prob in token_mid_plus_feature.items():
                            hovertext += f"<br>{token}: {prob:.2%}:"
                        hovertexts.extend([
                            "",  # start point
                            hovertext, 
                            None
                        ])
                    features[int(layer_idx)] = features_layer

                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color="red", width=2),
                        hovertext=hovertexts,
                        hoverinfo="text",
                        name=f"Layer {layer_idx} features",
                        visible=True,
                        showlegend=True,
                    ))

                st.plotly_chart(fig)

                # Plot the cosine sim between the mid->post from layer to layer
                # E.g. Cosine sim from mid->post of layer 0 and 1 etc.
                # for layer_idx, _ in st.session_state.gemma_residual_post_no_steering.items():
                #     if layer_idx == 0: # Layer 0 has no resid values behind it so continue
                #         continue
                #     r_mid, r_post = st.session_state.gemma_residual_mid_no_steering[layer_idx], st.session_state.gemma_residual_post_no_steering[layer_idx]
                #     r_mid_prev, r_post_prev = st.session_state.gemma_residual_mid_no_steering[layer_idx-1], st.session_state.gemma_residual_post_no_steering[layer_idx-1]
                    # r_direction = r_post - r_mid # Shape [1, seq_len, 2304]
                    # r_direction_prev = r_post_prev - r_mid_prev
                    # print(r_direction.shape)
                    # similarity_score = cosine_similarity([r_direction[0,-1,:].cpu().numpy()], [r_direction_prev[0,-1,:].cpu().numpy()])
                    # print(f"CoSim btw {layer_idx-1} and {layer_idx} is {similarity_score}")

                # Plot the KL divergence between the prob distribution
                fig2 = go.Figure()
                kl_div = []
                for layer_idx, _ in st.session_state.gemma_residual_post_no_steering.items():
                    if layer_idx == 0: # Layer 0 has no resid values behind it so continue
                        continue
                    r_mid, r_post = st.session_state.gemma_residual_mid_no_steering[layer_idx], st.session_state.gemma_residual_post_no_steering[layer_idx]

                    activations_norm_ln_final=gemma_2_2b.ln_final(r_mid)
                    my_logits = gemma_2_2b.unembed(activations_norm_ln_final)
                    my_logits_softcap = gemma_2_2b.cfg.output_logits_soft_cap * F.tanh(my_logits / gemma_2_2b.cfg.output_logits_soft_cap)
                    probs_mid = torch.softmax(my_logits_softcap[0, -1, :], dim=-1)

                    activations_norm_ln_final=gemma_2_2b.ln_final(r_post)
                    my_logits = gemma_2_2b.unembed(activations_norm_ln_final)
                    my_logits_softcap = gemma_2_2b.cfg.output_logits_soft_cap * F.tanh(my_logits / gemma_2_2b.cfg.output_logits_soft_cap)
                    probs_post = torch.softmax(my_logits_softcap[0, -1, :], dim=-1)

                    kl = F.kl_div(
                        probs_post.log(),
                        probs_mid,
                        reduction='sum'
                    )
                    kl_div.append(kl.item())
                fig2.add_trace(go.Scatter(
                    x=[i+1 for i in range(len(kl_div))],
                    y=kl_div,
                    mode='lines+markers',  # 'lines', 'markers', or 'lines+markers'
                    name='KL Divergence between mid and post for each layer',
                    line=dict(color='firebrick', width=4)
                ))
                st.plotly_chart(fig2)
                    
                cols = st.columns(3)
                keys = ["input_col1", "input_col2", "input_col3"]

                for i, col in enumerate(cols):
                    with col:
                        answer = st.text_input("Enter token:", key=keys[i])
                        if answer:
                            df = get_rank_data(answer, gemma_2_2b)
                            
                            # Apply styling to the rank columns
                            styled_df = df.style.applymap(color_ranks, subset=["Mid Rank", "Post Rank"])
                            
                            # Display with use_container_width to fit the column
                            table_height = (len(df) + 1) * 35 + 3
                            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=table_height)

                # Cosine sim between each feature
                # for layer_idx, j in features.items():
                #     l = []
                #     latent_idx_list = []
                #     for latent_idx, f in j.items():
                #         l.append(f.cpu().numpy())
                #         latent_idx_list.append(latent_idx)
                #     similarity_matrix = cosine_similarity(l)
                #     st.write(latent_idx_list)
                #     st.write(similarity_matrix)
                #     st.write("----------------")

        with st.expander("Features"):
            st.markdown("```\nClick on a mlp node to display the features extracted by the SAE\n```")
            if updated_state.selected_id is not None and updated_state.selected_id.startswith("mlp_"):
                layer_idx = int(updated_state.selected_id.split("_")[1])
                st.markdown(f"```\nSelected MLP Layer: {layer_idx}\n```")
                sae_act = st.session_state.gemma_sae_acts_no_steering[layer_idx]
                top_indices = np.argsort(np.abs(sae_act.numpy()[0, -1]))[-10:][::-1]
                top_indices_strength = sae_act.numpy()[0, -1][top_indices]
                st.write(f"```\nSparse feature vectors for MLP Layer {layer_idx}\n```")
                fig, ax = plt.subplots()
                ax.imshow(sae_act[0, -1, :].reshape(128, 128).detach().cpu().numpy())
                ax.axis("off")
                st.pyplot(fig)
                for latent_idx, strength in zip(top_indices, top_indices_strength):
                    with st.expander(f"Feature {latent_idx} with strength {strength:.3f}"):
                        threshold = gemma_2_2b_all_sae[layer_idx].threshold[latent_idx]
                        if strength < threshold:
                            continue
                        feature_direction = (
                            (strength - threshold).detach()
                            * gemma_2_2b_all_sae[layer_idx].W_dec[latent_idx]
                                .detach()
                        )
                        fig, ax = plt.subplots()
                        ax.imshow(feature_direction.reshape(48, 48).cpu().numpy())
                        ax.axis("off")
                        st.pyplot(fig)


    with col2:
        with st.form("steer_form"):
            st.markdown("```\nSteer the output of Gemma using\nthis interface.\n```")
            prompt_steered = st.text_input(
                "Prompt", 
                value=st.session_state.messages[0]["content"] if len(st.session_state.messages) > 0 else ""
                )
            layer = st.selectbox("Layer", range(26))
            feature = st.number_input("Feature", 0)
            strength = st.slider("Steering strength",-200.0, 200.0, 0.0,)
            run = st.form_submit_button("Run")

            if run:
                steering_args={"layer_idx":layer, "latent_idx":feature, "steering_coefficient":strength}
                gemma_2_2b, gemma_2_2b_all_sae = instantiate_model_gemma()
                with st.spinner("Running inference on steered model..."):
                    output, residual_mid, residual_post, captured_sae_acts_post, captured_sae_recons = run_gemma_with_hook_with_steering(gemma_2_2b, gemma_2_2b_all_sae, prompt_steered, steering_args={"layer_idx":layer, "latent_idx":feature, "steering_coefficient":strength})
                    st.session_state.gemma_output_steering = output
                    st.session_state.gemma_residual_mid_steering = residual_mid
                    st.session_state.gemma_residual_post_steering = residual_post
                    st.session_state.gemma_sae_acts_steering = captured_sae_acts_post
                    st.session_state.gemma_sae_recons_steering = captured_sae_recons
                    st.rerun()
                

with tab4:
    st.write("Misc")

