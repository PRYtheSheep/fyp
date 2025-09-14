import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

func_to_enable_grad = '_sample'
setattr(LlavaForConditionalGeneration, func_to_enable_grad, torch.enable_grad(getattr(LlavaForConditionalGeneration, func_to_enable_grad)))

model_id = "llava-hf/llava-1.5-7b-hf"
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

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 

async def forward_pass(image_path, prompt):

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
        max_new_tokens=200, 
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

    # print(processor.decode(output.sequences[0], skip_special_tokens=True))
    return output