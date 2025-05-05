"The full process of Long-Term Memory Question-Answering."

import json
import os
from container.memory_container import Round, Session, Conversation
from llms.packaged_llms import UnifiedLLM, get_messages
from reader.reader import PlainReader, CoNReader
from retriever.retriever import Retriever
from judge.llm_as_judge import LLMJudge
from termcolor import colored
from utils.chunks import integrate_same_sessions, reorganize_evidence_sessions, session2context
from utils.dates import date2datetime
... # Maybe more modules in the future

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Stage 1: Load Data
print("-"*40 + " Stage 1: Load Data " +"-"*40)
# data_path = "../data/longmemeval_oracle"
data_path = "../data/longmemeval_s"
with open(data_path, "r") as f:
    longmemeval_data = json.load(f)
print(f"Dataset(path: {data_path}) is loaded. Total number of samples: {len(longmemeval_data)}.")



# Stage 2: Preprocess Data
# Take the first sample as an example
print("-"*40 + " Stage 2: Preprocess Data " +"-"*40)

larger_llm_name = "Qwen2.5-14B-Instruct"
print(f"Loading Larger LLM ({larger_llm_name})...")
larger_llm = UnifiedLLM(larger_llm_name)
smaller_llm_name = "Qwen2.5-3B-Instruct"
print(f"Loading Larger LLM ({smaller_llm_name})...")
smaller_llm = UnifiedLLM(smaller_llm_name)


item = None
for tmp_item in longmemeval_data:
    if tmp_item['question_id'] == "118b2229":
        item = tmp_item
        break

print("Initializing Conversation...")
tmp_conversation = Conversation(
    sessions=item['haystack_sessions'],
    session_id_list=item['haystack_session_ids'],
    session_date_list=item['haystack_dates'],
    date=item['question_date'],
    id=item['question_id'],
    extract_keys=False,
    extract_facts=False,
    llm_extractor=larger_llm,
)

for session in tmp_conversation.sessions:
    session.extract_session_facts(larger_llm)

print(tmp_conversation)
for session in tmp_conversation.sessions:
    print(session)


# Stage 3: Retrieval
print("-"*40 + " Stage 3: Retrieval " +"-"*40)
retriever = Retriever()
current_question = item["question"]
current_answer = item["answer"]
question_date = item["question_date"]

retriever.compute_emb_for_conversation(tmp_conversation, strategy='session_facts', server='openai')
print("Emb computation finished...")
retriever.compute_scores_for_conversation(current_question, tmp_conversation,)
print("Score computation finished...")
top_k_facts, top_k_scores, top_k_sids = retriever.get_top_k(tmp_conversation,)

for i in range(len(top_k_scores)):
    print(f"{top_k_scores[i]}\t{top_k_sids[i]}\t{top_k_facts[i]}")



# Stage 4: Read and Answer
print("-"*40 + " Stage 4: Read and Answer " +"-"*40)
integrated_sids = integrate_same_sessions(top_k_sids, num = 5)
my_evidence_sessions = []
for sid in integrated_sids:
    for session in tmp_conversation.sessions:
        if session.session_id == sid:
            my_evidence_sessions.append(session)
            break
sorted_sessions = reorganize_evidence_sessions(my_evidence_sessions,)


print("Loading Reader...")
reader = PlainReader(reader_llm=larger_llm)
system_answer = reader.get_answer("session", sorted_sessions, current_question, question_date)

print("Loading LLM Judge...")
judge = LLMJudge(llm_judge=smaller_llm)
judgement = judge.judge(current_question, current_answer, system_answer)


print("-"*100)
print(f"QUESTION   : {current_question}")
print(f"GT ANSWER  : {current_answer}")
print(f"SYS ANSWER : {system_answer}")
print("-"*100)
print(f"LLM JUDGE  : \n{judgement}")
print("-"*100)