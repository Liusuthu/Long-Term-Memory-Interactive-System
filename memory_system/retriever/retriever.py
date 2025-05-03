import numpy as np
from openai import OpenAI
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from container.memory_container import Conversation
import os

os.environ["http_proxy"] = "http://127.0.0.1:37890"
os.environ["https_proxy"] = "http://127.0.0.1:37890"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
zhipu_api_key = os.getenv("ZHIPU_API_KEY")
client = OpenAI(api_key=api_key)
zhipu_client = ZhipuAI(api_key=zhipu_api_key)



class Retriever():
    "Contains all retrieval-related functions. Used as a integrated function base."
    def __init__(self,):
        self.openai_client = client
        self.zhipu_clietn = zhipu_client


    def get_embedding(self, text, model="text-embedding-ada-002"):
        """
        Args:
            model (str): Choose from `text-embedding-3-small`(1536), `text-embedding-3-large`(3072), `text-embedding-ada-002`(1536).
        """
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model=model).data[0].embedding

    def get_zhipu_embedding(self, text, model="embedding-2"):
        """
        Args:
            model (str): Choose from `embedding-3`(2048), `embedding-2`(1024).
        """
        text = text.replace("\n", " ")
        return self.zhipu_client.embeddings.create(input = [text], model=model).data[0].embedding


    def question2query(self, question):
        ...


    def compute_embed_for_conversation(self, conversation:Conversation, strategy:str="session"):
        "要设计好不同的strategy下计算什么embedding, 以便后面检索"
        ...


    def compute_similarity(emb1, emb2):
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        return np.dot(emb1, emb2) / (norm_emb1 * norm_emb2)


    def get_top_k(self, chunks, k=5, strategy:str="session"):
        ...