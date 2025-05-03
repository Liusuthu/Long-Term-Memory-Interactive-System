"Load Embedding Models To Implement Retrieval"

from openai import OpenAI
from zhipuai import ZhipuAI
from dotenv import load_dotenv
import os 


os.environ["http_proxy"] = "http://127.0.0.1:37890"
os.environ["https_proxy"] = "http://127.0.0.1:37890"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
zhipu_api_key = os.getenv("ZHIPU_API_KEY")
client = OpenAI(api_key=api_key)
zhipu_client = ZhipuAI(api_key=zhipu_api_key)


def get_embedding(text, model="text-embedding-3-small"):
    """
    Args:
        model (str): Choose from `text-embedding-3-small`(1536), `text-embedding-3-large`(3072), `text-embedding-ada-002`().
    """
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


def get_zhipu_embedding(text, model="embedding-2"):
    """
    Args:
        model (str): Choose from `embedding-3`(2048), `embedding-2`(1024).
    """
    text = text.replace("\n", " ")
    return zhipu_client.embeddings.create(input = [text], model=model).data[0].embedding


# ZHIPU充了2块钱，不知道能用多久 后续还可以引入更多，今天先不了。
# print("OpenAI Emb:",(len(get_embedding(text="good morning", model="text-embedding-3-large"))))
# print("ZHIPU Embed:", (len(get_zhipu_embedding(text="good morning"))))