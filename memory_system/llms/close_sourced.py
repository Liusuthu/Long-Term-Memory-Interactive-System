"Use LLMs Via APIs"
# TODO: OpenAI / Claude / Gemini

from openai import OpenAI
from dotenv import load_dotenv
import os 


os.environ["http_proxy"] = "http://127.0.0.1:37890"
os.environ["https_proxy"] = "http://127.0.0.1:37890"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def get_client():
    client = OpenAI(api_key=api_key)
    return client


def get_response(client, messages, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content



# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful, detailed, and polite AI assistant." },
#         {"role": "user", "content": "Introduce yourself."},
#     ],
#     temperature=0.7,
# )

# model_answer = response.choices[0].message.content
# print(model_answer)