from retriever.get_embed import get_embedding, get_zhipu_embedding
import numpy as np


def compute_similarity(emb1, emb2):
    norm_emb1 = np.linalg.norm(emb1)
    norm_emb2 = np.linalg.norm(emb2)
    return np.dot(emb1, emb2) / (norm_emb1 * norm_emb2)


print(compute_similarity(
    get_embedding("如何衡量我们之间的关系"),
    get_embedding("如何衡量我们之间的关联")
))