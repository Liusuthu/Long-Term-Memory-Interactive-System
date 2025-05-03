from llms.open_sourced import load_llm_model, load_qwen3_model, llm_generate, qwen3_generate
from llms.close_sourced import get_client, get_response

# export PYTHONPATH=$PYTHONPATH:/home/limusheng/Long-Term-Memory-Interactive-System

class PlainReader():
    "Simply use the retrieved text as input"
    def __init__(self, model_name="Qwen2.5-7B-Instruct"):
        self.reader_tokenizer = None
        self.reader_llm = None
        self.client = None
        self.model_name = model_name
        if ("Qwen2" in model_name) or ("Llama" in model_name) or ("Mistral" in model_name):
            self.reader_tokenizer, self.reader_llm = load_llm_model(model_name)
            self.generate_function = llm_generate
        elif "Qwen3" in model_name:
            self.reader_tokenizer, self.reader_llm = load_qwen3_model(model_name)
            self.generate_function = qwen3_generate
        elif "gpt" in model_name:
            self.client = get_client()
            self.generate_function = get_response
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
    
    def get_answer(self, retrieval_type, retrieval_list):
        """
        The function used to generate answer from retrieved chunks.
        Args:
            retrieval_type (str): Choose from `round`, `session` or `hybrid`.
            retrieval_list (list): Retrieved chunks, organized in a list.
        """
        if retrieval_type == "round":
            ...

        elif retrieval_type == "session":
            ...
        
        elif retrieval_type == "hybrid":
            ...

        else:
            raise ValueError(f"Argument retrieval_type should be one of `round`, `session` or `hybrid`, but now is {retrieval_type}.")
        


class CoNReader():
    "First filter the retrieved chunks, then get answer."
    def __init__(self, reader_model_name="Qwen2.5-7B-Instruct", filter_model_name="Qwen2.5-3B-Instruct"):
        self.reader_tokenizer = None
        self.reader_llm = None
        self.filter_tokenizer = None
        self.filter_llm =None
        self.client = None
        self.reader_model_name = reader_model_name
        self.filter_model_name = filter_model_name
        
        # Load reader model
        if ("Qwen2" in reader_model_name) or ("Llama" in reader_model_name) or ("Mistral" in reader_model_name):
            self.reader_tokenizer, self.reader_llm = load_llm_model(reader_model_name)
            self.reader_generate_function = llm_generate
        elif "Qwen3" in reader_model_name:
            self.reader_tokenizer, self.reader_llm = load_qwen3_model(reader_model_name)
            self.reader_generate_function = qwen3_generate
        elif "gpt" in reader_model_name:
            self.client = get_client()
            self.reader_generate_function = get_response
        else:
            raise ValueError(f"Unsupported reader model: {reader_model_name}")
        
        # Load filter model
        if ("Qwen2" in filter_model_name) or ("Llama" in filter_model_name) or ("Mistral" in filter_model_name):
            self.filter_tokenizer, self.filter_llm = load_llm_model(filter_model_name)
            self.filter_generate_function = llm_generate
        elif "Qwen3" in filter_model_name:
            self.filter_tokenizer, self.filter_llm = load_qwen3_model(filter_model_name)
            self.filter_generate_function = qwen3_generate
        elif "gpt" in filter_model_name:
            self.client = get_client()
            self.filter_generate_function = get_response
        else:
            raise ValueError(f"Unsupported filter model: {filter_model_name}")


    def get_answer(self, retrieval_type, retrieval_list):
        """
        The function used to generate answer from retrieved chunks.
        Args:
            retrieval_type (str): Choose from `round`, `session` or `hybrid`.
            retrieval_list (list): Retrieved chunks, organized in a list.
        """
        if retrieval_type == "round":
            ...

        elif retrieval_type == "session":
            ...
        
        elif retrieval_type == "hybrid":
            ...

        else:
            raise ValueError(f"Argument retrieval_type should be one of `round`, `session` or `hybrid`, but now is {retrieval_type}.")




prompt = "A+B=3, B+C=5, known that A=1, C=?."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
# plain_reader = PlainReader()
# result = plain_reader.generate_function(plain_reader.reader_tokenizer, plain_reader.reader_llm, messages)

gpt_reader = PlainReader(model_name='gpt-4o-mini')
result = gpt_reader.generate_function(gpt_reader.client,messages)

print(result)

