import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

func_to_enable_grad = '_sample'
setattr(LlavaForConditionalGeneration, func_to_enable_grad, torch.enable_grad(getattr(LlavaForConditionalGeneration, func_to_enable_grad)))

# Use absolute path
vit_attn_folder = r"C:\Users\Dreamcore\OneDrive\Desktop\fyp\saved\vit_attn"
generated_folder = r"C:\Users\Dreamcore\OneDrive\Desktop\fyp\saved\generated"

model_id = "llava-hf/llava-1.5-7b-hf"

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
        attn_implementation = "eager"
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