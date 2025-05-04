"The full process of Long-Term Memory Question-Answering."

import json
from container.memory_container import Round, Session, Conversation
from llms.packaged_llms import UnifiedLLM, get_messages
from reader.reader import PlainReader, CoNReader
from retriever.retriever import Retriever
from termcolor import colored
... # Maybe more modules in the future


# Stage 1: Load Data
print("-"*40 + " Stage 1: Load Data " +"-"*40)
# data_path = "../data/longmemeval_oracle"
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

item = None
for tmp_item in longmemeval_data:
    if tmp_item['question_id'] == "gpt4_2655b836":
        item = tmp_item
        break

# item = longmemeval_data[0]

print("Initializing Conversation...")
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

for session in tmp_conversation.sessions:
    session.extract_session_facts(llm_extractor)


print(tmp_conversation)

for session in tmp_conversation.sessions:
    print(session)

# Then compute embedding for some structures.



# Stage 3: Retrieval
print("-"*40 + " Stage 3: Retrieval " +"-"*40)
retriever = Retriever()
current_question = item["question"]
current_answer = item["answer"]
print(f"Expected QA:\n  QUESTION: {current_question}\n  ANSWER: {current_answer}")

retriever.compute_emb_for_conversation(tmp_conversation,)
print("Emb computation finished...")
retriever.compute_scores_for_conversation(current_question, tmp_conversation,)
print("Score computation finished...")
top_k_facts, top_k_scores, top_k_sids = retriever.get_top_k(tmp_conversation,)

for i in range(len(top_k_scores)):
    print(f"{top_k_scores[i]}\t{top_k_sids[i]}\t{top_k_facts[i]}")