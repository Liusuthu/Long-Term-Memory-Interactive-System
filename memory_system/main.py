"The full process of Long-Term Memory Question-Answering."

import json
from container.memory_container import Round, Session, Conversation
from llms.packaged_llms import UnifiedLLM
from reader.reader import PlainReader, CoNReader
from retriever.get_embed import get_embedding, get_zhipu_embedding
... # Maybe more modules in the future


# Stage 1: Load Data
print("-"*40 + " Stage 1: Load Data " +"-"*40)
data_path = "../data/longmemeval_oracle"
with open(data_path, "r") as f:
    longmemeval_data = json.load(f)
print(f"Dataset(path: {data_path}) is loaded. Total number of samples: {len(longmemeval_data)}.")



# Stage 2: Preprocess Data
# Take the first sample as an example
print("-"*40 + " Stage 2: Preprocess Data " +"-"*40)

extractor_name = "Qwen2.5-14B-Instruct"
print(f"Loading LLM Extractor({extractor_name})...")
llm_extractor = UnifiedLLM(extractor_name)

item = longmemeval_data[0]
tmp_conversation = Conversation(
    sessions=item['haystack_sessions'],
    session_id_list=item['haystack_session_ids'],
    session_date_list=item['haystack_dates'],
    date=item['question_date'],
    id=item['question_id'],
    extract_keys=False,
    extract_facts=False,
    llm_extractor=llm_extractor,
)
print(tmp_conversation)

# Then compute embedding for some structures.



# Stage 3: Retrieval
current_question = item["question"]
current_answer = item["answer"]
print(f"Expected QA:\n  QUESTION: {current_question}\n  ANSWER: {current_answer}")