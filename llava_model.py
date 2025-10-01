import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
from transformers import BitsAndBytesConfig
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

func_to_enable_grad = '_sample'
setattr(LlavaForConditionalGeneration, func_to_enable_grad, torch.enable_grad(getattr(LlavaForConditionalGeneration, func_to_enable_grad)))

# Use absolute path
save_folder = r"C:\Users\Dreamcore\OneDrive\Desktop\fyp\saved"
vit_attn_folder = os.path.join(save_folder, "vit_attn")
generated_folder = os.path.join(save_folder, "generated")
attn_folder = os.path.join(save_folder, "attn")

model_id = "llava-hf/llava-1.5-7b-hf"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    offload_state_dict=True
)

def get_processor():
    return AutoProcessor.from_pretrained(model_id)

def instantiate_model():
    """
    Instantiates a LLaVa model and returns the model, processor and hooks. Only instantiate 1 model at a time due to GPU memory limits.
    Use del to delete the model, processor and hooks before instantiating a new model
    """
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        quantization_config=quant_config,
        attn_implementation = "eager",
    ).to(0)

    #--------------------------------------------------
    model.vision_tower.config.output_attentions = True

    # set hooks to get attention weights
    model.enc_attn_weights = []
    #outputs: attn_output, attn_weights, past_key_value
    def forward_hook(module, inputs, output): 
        if output[1] is None:
            logger.error(
                ("Attention weights were not returned for the encoder. "
                "To enable, set output_attentions=True in the forward pass of the model. ")
            )
            return output
        
        output[1].requires_grad_(True)
        output[1].retain_grad()
        model.enc_attn_weights.append(output[1].detach().cpu())
        return output

    hooks_pre_encoder, hooks_encoder = [], []
    for layer in model.language_model.layers:
        hook_encoder_layer = layer.self_attn.register_forward_hook(forward_hook)
        hooks_pre_encoder.append(hook_encoder_layer)

    model.enc_attn_weights_vit = []

    def forward_hook_image_processor(module, inputs, output): 
        if output[1] is None:
            logger.warning(
                ("Attention weights were not returned for the vision model. "
                "Relevancy maps will not be calculated for the vision model. " 
                "To enable, set output_attentions=True in the forward pass of vision_tower. ")
            )
            return output

        output[1].requires_grad_(True)
        output[1].retain_grad()
        model.enc_attn_weights_vit.append(output[1])
        return output

    hooks_pre_encoder_vit = []
    for layer in model.vision_tower.vision_model.encoder.layers:
        hook_encoder_layer_vit = layer.self_attn.register_forward_hook(forward_hook_image_processor)
        hooks_pre_encoder_vit.append(hook_encoder_layer_vit)
    #--------------------------------------------------

    processor = AutoProcessor.from_pretrained(model_id)

    if model.language_model.config.model_type == "gemma":
        eos_token_id = processor.tokenizer('<end_of_turn>', add_special_tokens=False).input_ids[0]
    else:
        eos_token_id = processor.tokenizer.eos_token_id

    return model, processor, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id

def forward_pass(model, processor, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id, image_path, prompt):
    """
    Run a forward passwith model.generate()
    """

    # Save the original prompt before it gets yoinked by the processor
    file_path = os.path.join(generated_folder, "original_prompt.txt")
    with open(file_path, "w") as f:
        f.write(prompt)

    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    output = model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=False,
        use_cache=True,
        output_attentions=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        output_scores=True,
        eos_token_id=eos_token_id
    )

    for h in hooks_pre_encoder:
        h.remove()
    for h in hooks_pre_encoder_vit:
        h.remove()

    # Save the vit attention weights
    for i, attn in enumerate(model.enc_attn_weights_vit):
        file_path = os.path.join(vit_attn_folder, f"{i}.pt")
        torch.save(attn, file_path)

    # Save the output sequence
    file_path = os.path.join(generated_folder, "generated.pt")
    torch.save(output.sequences, file_path)

    num_forward_pass = int(len(model.enc_attn_weights)/32)
    file_path = os.path.join(generated_folder, "num.txt")
    with open(file_path, "w") as f:
        f.write(str(num_forward_pass))

    return output

def forward_pass_one_step(model, processor, hooks_pre_encoder, hooks_pre_encoder_vit, eos_token_id, image_path, prompt, assistant_prompt=None):
    """
    Run a forward passwith model()
    """
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image"},
            ],
        },
    ]
    # Append assistant prompt into conversation if any
    if assistant_prompt is not None:
        conversation += [
            {"role": "assistant", "content": [{"type": "text", "text": assistant_prompt}]}
        ]
    print(conversation)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=False, return_tensors="pt")
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    decoded_input_tokens = [processor.decode([t]) for t in inputs["input_ids"][0]]
    for i, tok in enumerate(decoded_input_tokens):
        print(f"{i}: {repr(tok)}")

    output = model(**inputs,
                    use_cache=False,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    eos_token_id=eos_token_id)
    
    for h in hooks_pre_encoder:
        h.remove()
    for h in hooks_pre_encoder_vit:
        h.remove()

    # Save the model attn weights for testing
    for i, attn in enumerate(model.enc_attn_weights):
        file_path = os.path.join(attn_folder, f"{i}.pt")
        torch.save(attn, file_path)

    return output

def attention_rollout(attn_maps):
    """
    Performs rollout on the provided attention maps. Pass in the model attn weights e.g. model.enc_attn_weights[0:32]
    """
    attn_rollout = []
    device = attn_maps[0].device
    batch_size, _, seq_len, _ = attn_maps[0].shape
    
    # Identity matrix for self-attention
    I = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len, seq_len)

    prod = I.clone()
    
    for i, attn_map in enumerate(attn_maps):
        # Average over heads → [batch, seq_len, seq_len]
        attn_map = attn_map.mean(dim=1)
        
        # Add identity and multiply
        prod = prod @ (attn_map + I)
        
        # Normalize across sequence dimension
        prod = prod / prod.sum(dim=-1, keepdim=True)
        
        attn_rollout.append(prod)

    return attn_rollout

def get_important_tokens(attn_map, raw_image, n=50):
    """
    Get the important image and text tokens from the attention map, either from attenion rollouts or the raw attention map itself
    attn_map is a single attention layer from model.enc_attn_weights or from a single player from the attention rollout

    Assumes that the input image is larger than 24x24 as LLaVa resizes images larger than 24x24 into 24x24. Not sure about images
    smaller than 24x24
    """
    token_importances = attn_map[0, -2]  # importance of all tokens for position t
    values, indices = torch.topk(token_importances, k=n)

    impt_text_tokens_index = []
    heatmap_raw = [0 for i in range(576)]
    for score, idx in zip(values.tolist(), indices.tolist()):
        if idx >= 5 and idx <= 580:
            # Image tokens
            heatmap_raw[idx-5] = 1
        else:
            impt_text_tokens_index.append(idx)

    cls_attn = torch.tensor(heatmap_raw)
    H, W = 24, 24  # 336 / 14
    heatmap = cls_attn.reshape(H, W).detach().cpu().numpy()

    # Assuming heatmap shape [H, W]
    heatmap_tensor = torch.tensor(heatmap[None, None], dtype=torch.float32)
    heatmap_full = F.interpolate(heatmap_tensor, size=(raw_image.size[1], raw_image.size[0]), mode='nearest')[0,0].numpy()

    return heatmap_full, impt_text_tokens_index