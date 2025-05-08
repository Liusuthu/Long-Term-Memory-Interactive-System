import torch
import torch.nn.functional as F
import os 

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0")

model_path = "/nas/datasets/zxiao28/llm/gte-Qwen2-1.5B-instruct"
print("Loading GTE Model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
gte_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
gte_model.eval()
max_length = 8192


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_qwen_embedding(query_batch, tokenizer=tokenizer, model=gte_model, max_length=max_length):
    with torch.no_grad():  # ✅ 防止构建计算图，避免显存泄漏
        batch_dict = tokenizer(query_batch, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

