from open_sourced import load_llm_model, load_qwen3_model, llm_generate, qwen3_generate
from close_sourced import get_client, get_response


# Open
# tknz, llm = load_llm_model("Qwen2.5-7B-Instruct")

# prompt = "A+B=3, B+C=5, known that A=1, C=?."
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]

# print(llm_generate(tknz,llm,messages))



# Test Closed 
# prompt = "A+B=3, B+C=5, known that A=1, C=?."
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]

# client = get_client()
# print(get_response(client, messages))