"Integrated Open-Sourced and Close-Sourced LLMs into a unified class, only with a unified method generate(messages).\n As well as a function get_messages() to get messages."

from open_sourced import load_llm_model, load_qwen3_model, llm_generate, qwen3_generate
from close_sourced import get_client, get_response


# Message-Formatter
def get_messages(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages


class UnifiedLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.llm = None
        self.client = None
        if ('Qwen2' in model_name) or ('Llama' in model_name) or ('Mistral' in model_name):
            self.tokenizer, self.llm = load_llm_model(model_name)
        elif 'Qwen3' in model_name:
            self.tokenizer, self.llm = load_qwen3_model(model_name)
        elif 'gpt' in model_name:
            self.client = get_client()
        else:
            raise ValueError(f"Model {model_name} is not yet support, please check carefully.")
        
    def generate(self, messages):
        "A unified generate function for all LLMs."
        if ('Qwen2' in self.model_name) or ('Llama' in self.model_name) or ('Mistral' in self.model_name):
            return llm_generate(self.tokenizer, self.llm, messages)
        elif 'Qwen3' in self.model_name:
            return qwen3_generate(self.tokenizer, self.llm, messages)
        elif 'gpt' in self.model_name:
            return get_response(self.client, messages, self.model_name)
        else:
            raise ValueError(f"Model {self.model_name} is not yet support, please check carefully.")
    

    # Print the LLM
    def __repr__(self):
        if 'gpt' in self.model_name:
            return f"OpenAILLM(\n  model_name={self.model_name}, \n  client={self.client},\n)"
        else:
            return f"CasualLLM(\n  model_name={self.model_name}, \n  tokenizer={self.tokenizer}, \n  llm={self.llm}, \n)"




# gpt_model = UnifiedLLM('gpt-4o-mini')
# qwen2_model = UnifiedLLM('Qwen2.5-7B-Instruct')
# llama_model = UnifiedLLM('Llama-3.2-1B-Instruct')

# print(gpt_model)
# print(qwen2_model)
# print(llama_model)

# prompt = "A+B=3, B+C=5, known that A=1, C=?."
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]

# print(gpt_model.generate(messages))
# print("-"*100)
# print(qwen2_model.generate(messages))
# print(llama_model.generate(messages))