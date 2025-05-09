"Use Open-Sourced LLMs"
# TODO: Qwen, LLaMA, DeepSeek.

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch

# ----------------------------- Load Model ----------------------------- #
# Qwen2 / LLaMA3 / Mistral
def load_llm_model(model_name):
    model_path = "/nas/datasets/zxiao28/llm/"+model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map={"": "cuda:0"},
    )
    return tokenizer, model


# Qwen3
def load_qwen3_model(model_name):
    model_path = "/nas/datasets/zxiao28/llm/Qwen/"+model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map={"": "cuda:0"},
    )
    return tokenizer, model




# ----------------------------- Generate ----------------------------- #
# Qwen2 / LLaMA3 / Mistral
def llm_generate(tokenizer, model, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# Qwen3
def qwen3_generate(tokenizer, model, messages, thinking=True):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking 
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # conduct text completion
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    # parsing thinking content
    if thinking:
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return thinking_content, content
    else:
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return "", content



# tok, llm = load_llm_model("Llama-3.2-1B-Instruct")

# prompt = "A+B=3, B+C=5, known that A=1, C=?."
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]

# print(llm_generate(tok,llm,messages))