import numpy as np
from openai import OpenAI
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from container.memory_container import Conversation
import os
import heapq


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


    def get_openai_embedding(self, text, model="text-embedding-ada-002"):
        """
        Args:
            model (str): Choose from `text-embedding-3-small`(1536), `text-embedding-3-large`(3072), `text-embedding-ada-002`(1536).
        """
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(input = [text], model=model).data[0].embedding

    def get_zhipu_embedding(self, text, model="embedding-2"):
        """
        Args:
            model (str): Choose from `embedding-3`(2048), `embedding-2`(1024).
        """
        text = text.replace("\n", " ")
        return self.zhipu_client.embeddings.create(input = [text], model=model).data[0].embedding


    def question2query(self, question):
        ...


    def compute_emb_for_conversation(self, conversation:Conversation, strategy:str="session_facts", server:str="openai"):
        "要设计好不同的strategy下计算什么embedding, 以便后面检索"
        if strategy=="session_facts":
            if server=="openai":
                for session in conversation.sessions:
                    session.session_facts_emb = []
                    for fact in session.session_facts:
                        emb = self.get_openai_embedding(fact)
                        session.session_facts_emb.append(emb)
            elif server=="zhipu":
                for session in conversation.sessions:
                    session.session_facts_emb = []
                    for fact in session.session_facts:
                        emb = self.get_zhipu_embedding(fact)
                        session.session_facts_emb.append(emb)
            else:
                ValueError(f"Embedding server {server} not supported yet.")
        else:
            raise ValueError(f"Compute Embedding Stragtegy {strategy} not supported yet.")


    def compute_scores_for_conversation(self, query, conversation:Conversation, strategy:str="session_facts", server:str="openai"):
        if strategy=="session_facts":
            if server=="openai":
                query_emb = self.get_openai_embedding(query)
                for session in conversation.sessions:
                    session.session_facts_scores = []
                    for fact_emb in session.session_facts_emb:
                        session.session_facts_scores.append(self.compute_similarity(fact_emb, query_emb))
            elif server=="zhipu":
                query_emb = self.get_zhipu_embedding(query)
                for session in conversation.sessions:
                    session.session_facts_scores = []
                    for fact_emb in session.session_facts_emb:
                        session.session_facts_scores.append(self.compute_similarity(fact_emb, query_emb))
            else:
                ValueError(f"Embedding server {server} not supported yet.")
        else:
            raise ValueError(f"Stragtegy {strategy} not supported yet.")


    def compute_similarity(self,emb1, emb2):
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        return np.dot(emb1, emb2) / (norm_emb1 * norm_emb2)



    def get_top_k(self, conversation, k=10, strategy:str="session_facts"):
        top_k_facts = []
        top_k_scores = []
        top_k_session_ids = []

        for session in conversation.sessions:
            session_scores = session.session_facts_scores
            session_facts = session.session_facts
            session_id = session.session_id

            session_fact_scores = [(fact, score, session_id) for fact, score in zip(session_facts, session_scores)]
            top_facts = heapq.nlargest(k, session_fact_scores, key=lambda x: x[1])

            for fact, score, id in top_facts:
                top_k_facts.append(fact)
                top_k_scores.append(score)
                top_k_session_ids.append(id)

        combined = list(zip(top_k_facts, top_k_scores, top_k_session_ids))
        # combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
        combined_sorted = heapq.nlargest(k, combined, key=lambda x: x[1]) # Global Top K

        top_k_facts_sorted, top_k_scores_sorted, top_k_session_ids_sorted = zip(*combined_sorted) if combined_sorted else ([], [], [])

        return list(top_k_facts_sorted), list(top_k_scores_sorted), list(top_k_session_ids_sorted)