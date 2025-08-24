
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)
import re
from tqdm import tqdm
import numpy as np
import base64
from io import BytesIO
import inspect

import util

def load_model(model_id, device='auto'):
    model_name = model_id.split('/')[1]
    if model_name.startswith('gpt-oss'):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype='auto', 
            device_map=device,
            trust_remote_code=True
            ).eval()
    elif model_name.startswith("pangu-pro-moe-model"):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            trust_remote_code=True
            ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            attn_implementation='flash_attention_2',
            trust_remote_code=True
            ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side  = 'left'
    if tokenizer.pad_token is None:
        # We can set pad_token as the eos_token or add a new one
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate_output(model, tokenizer, prompts, batch_size=8, max_new_tokens=1024, model_name="default"):
    all_outputs = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_prompts in tqdm(util.batchify(prompts, batch_size), total=total_batches):
        # Tokenize the batch
        input_tokens = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
            ).to(model.device)
        if "token_type_ids" in input_tokens:
            forward_args = inspect.signature(model.forward).parameters
            if "token_type_ids" not in forward_args:
                input_tokens.pop("token_type_ids")
        input_ids = input_tokens["input_ids"]

        output_ids = model.generate(**input_tokens, 
                                    max_new_tokens=max_new_tokens, 
                                    # do_sample=True, 
                                    return_dict_in_generate=True,
                                    max_length=max_new_tokens
                                    # temperature = 0.3,
                                    # attention_mask = input_tokens["attention_mask"]
                                    )
        # Extract only the generated tokens (excluding the input prompts)
        generated_tokens = [
            output[ids.shape[-1]:] for ids, output in zip(input_ids, output_ids["sequences"])
        ]

        # Decode the generated outputs
        # responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        responses = [clean_generated_text(tokenizer.decode(generated_tokens[i], skip_special_tokens=True)) for i in range(len(generated_tokens))]
        # print(responses)
        all_outputs.extend(responses)

    return all_outputs 

def moderate(model, tokenizer, prompt):
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def construct_prompt(tokenizer, model_name, questions, system_prompt=None, images=False, enable_thinking=False, responses=None):
    prompts = []
    if system_prompt is None:
        system_prompt = "You are an AI that provides direct and precise answers. Respond only to the question without additional details or explanations."
    # You are an AI designed to offer unbiased, direct, and honest answers.
    #  Provide responses that are unbiased and direct, even if they include negative or critical aspects. Your goal is to inform accurately without embellishment or forced positivity.
    if images:
        if model_name.startswith('gemma-3'):
            images = [encode_image_to_base64(example['image']) for example in images]
        elif model_name.startswith("Qwen2.5-VL"):
            images = [img["image"].filename for img in images]
    for i, question in enumerate(questions):
        if model_name.startswith('gemma-3') or model_name.startswith("Qwen2.5-VL"):
            if images:
                assert len(questions) == len(images), "Questions and images should have the same length."
                # Load and encode the image from your local directory.
                # with open(images[i], "rb") as image_file:
                    # encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                chat = [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user",
                        "content": [
                            {"type": "image", "image": images[i]},
                            {"type": "text", "text": question}
                        ]
                    }
                ]
            else:
                chat = [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user",
                        "content": [
                            {"type": "text", "text": question}
                        ]
                    }
                ]
        elif model_name.startswith('gemma') or model_name.startswith('Quill'):
            chat = [
                {"role": "user", "content": question}
            ]
        elif model_name.startswith('Llama-Guard'):
            if responses is None:
                raise Exception("Response is not provided for the prompt construction!")
            response = responses[i]
            if not response.strip():
                response = "Sorry, I cannot assist with that."
            response = extract_text_after_think(response)
            if model_name == "Llama-Guard-3-8B":
                chat = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": str(response).replace('[', '').replace(']', '')},
                ]
            else:
                raise Exception(f"Unknown Llama-Guard version ({model_name})")
        else:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        if model_name.startswith('gemma-3') or model_name.startswith("Qwen2.5-VL"):
            prompt = chat
        else:
            if model_name.startswith('Qwen3') or model_name=="Hunyuan-A13B-Instruct":
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            else:
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts

def construct_judge_prompt(tokenizer, questions, responses):
    prompts = []
    for question, response in zip(questions, responses):
        if not response.strip():
            response = "Sorry, I cannot assist with that."
        response = extract_text_after_think(response)
        chat = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": str(response).replace('[', '').replace(']', '')},
        ]
        # prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        prompts.append(chat)
    return prompts

def count_mlp_module(model, model_name):
    mlp_count = 0
    for name, module in model.named_modules():
        if any(keyword in name.lower() for keyword in ['gate', 'up']):
            # print(name)
            mlp_count += 1
    # These two model has gate and up fused into one single layer
    if model_name.lower().startswith("phi-4") or model_name.lower().startswith("dna"):
        return mlp_count
    else:
        return int(mlp_count/2)
    
def extract_text_after_think(response: str) -> str:
    # Find all occurrences of </think>
    think_matches = list(re.finditer(r"</think>", response))

    if think_matches:
        # Get the last occurrence
        last_think_index = think_matches[-1].end()
        return response[last_think_index:].lstrip()  # Strip leading spaces/newlines
    else:
        return response  # No </think> tag, return entire response

def encode_image_to_base64(image):
    # Convert the image to base64 encoding
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def clean_generated_text(text):
    """Cleans generated text by removing unwanted prefixes like 'Assistant:', '\n', or leading spaces."""
    # if model_name.startswith("DeepSeek"):
    # text = extract_text_after_think(text)
    text = text.strip()  # Remove leading/trailing whitespace or newlines
    text = re.sub(r"^(assistant\n|Assistant:|AI:|Bot:|Response:|Reply:|.:)\s*", "", text, flags=re.IGNORECASE)  # Remove AI labels if present
    text = text.strip()  # Remove leading/trailing whitespace or newlines
    # remove the answer tags if they exist
    answer_pattern = r'<answer>(.*?)</answer>'
    text = re.sub(answer_pattern, r'\1', text, flags=re.DOTALL)
    return text

def extract_code(text):
    # Regex to match content inside triple single quotes, ignoring an optional language specifier
    matches = re.findall(r"```\s*(\w+)?\s*\n(.*?)```", text, re.DOTALL)
    return '\n\n'.join(code for _, code in matches) if matches else "No code found."

def get_expert_usage_diff(activations, labels):
    # Stack all batch results into one array: shape (total_prompts, 128)
    # all_counts = np.concatenate(activations[layer_name], axis=0)  # shape: (num_prompts, 128)
    labels = np.array(labels)  # shape: (num_prompts,)
    assert activations.shape[0] == labels.shape[0], "Mismatch between prompt count and label count"
    # Split by label
    act_label_0 = activations[labels == 0]  # shape: (num_0_prompts, 128)
    act_label_1 = activations[labels == 1]  # shape: (num_1_prompts, 128)
    return act_label_0.mean(axis=0), act_label_1.mean(axis=0), act_label_0.mean(axis=0) - act_label_1.mean(axis=0)

def get_gate_layer(safety_experts, config, num_layers):
    layers = []
    if num_layers == 0:
        return layers
    for layer_name, _ in safety_experts.items():
        # get the layer index
        i = int(layer_name.split('.')[2])
        if i in range(num_layers):
            layers.append(f"model.layers.{i}.{config.name_gate}")
    return layers

def get_router_expert_layer(safety_experts, config, num_layers):
    layers = []
    if num_layers == 0:
        return layers
    if config.model_name == "deepseek-moe-16b-chat":
        start_idx = 1
    else:
        start_idx = 0
    for layer_name, expert_index in safety_experts.items():
        # get the layer index
        i = int(layer_name.split('.')[2])
        if i in range(start_idx, num_layers):
            if expert_index.any():
                if config.model_name == 'gpt-oss-20b':
                    layers.extend(
                        f"model.layers.{i}.{config.name_router_expert}.{target_name}.{idx}"
                        for idx in expert_index
                        for target_name in config.name_expert_layers
                    )
                else:
                    layers.extend(
                        f"model.layers.{i}.{config.name_router_expert}.{idx}.{target_name}"
                        for idx in expert_index
                        for target_name in config.name_expert_layers
                    )
    return layers

def get_shared_expert_layer(num_layers, model_config):
    layers = []
    if num_layers == 0:
        return layers
    if model_config.model_name == "deepseek-moe-16b-chat":
        layers += [f"model.layers.0.mlp.{name}" for name in model_config.name_expert_layers]
        start_idx = 1
    else:
        start_idx = 0
    for i in range(start_idx, num_layers):
        layers.extend(
            f"model.layers.{i}.{model_config.name_shared_expert}.{target_name}"
            for target_name in model_config.name_expert_layers
        )
    return layers