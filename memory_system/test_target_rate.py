"Test the retrieval target rate @k."

# 分为三种：完全没命中、部分命中、完全命中

import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from container.memory_container import Round, Session, Conversation
from llms.packaged_llms import UnifiedLLM, get_messages
from reader.reader import PlainReader, CoNReader
from retriever.retriever import Retriever
from judge.llm_as_judge import LLMJudge
from termcolor import colored
from utils.chunks import integrate_same_sessions, reorganize_evidence_sessions, session2context
from utils.dates import date2datetime
from utils.target import get_target
import torch
... # Maybe more modules in the future



# Stage 1: Load Data
print("-"*40 + " Stage 1: Load Data " +"-"*40)
# data_path = "../data/longmemeval_oracle"
data_path = "../data/longmemeval_s"
with open(data_path, "r") as f:
    longmemeval_data = json.load(f)
print(f"Dataset(path: {data_path}) is loaded. Total number of samples: {len(longmemeval_data)}.")



# Stage 2: Load Data and Start Retrieving
print("-"*40 + " Stage 2: Start testing " +"-"*40)
larger_llm_name = "Qwen2.5-14B-Instruct"
print(f"Loading Larger LLM ({larger_llm_name})...")
larger_llm = UnifiedLLM(larger_llm_name)
# smaller_llm_name = "Qwen2.5-3B-Instruct"
# print(f"Loading Smaller LLM ({smaller_llm_name})...")
# smaller_llm = UnifiedLLM(smaller_llm_name)


# 分不同问题类别进行测试命中情况(命中率)
question_type = ["single-session-preference"]
retriever = Retriever()
total_count = 0

fully_target_count = 0
partly_target_count = 0
no_target_count = 0

top5_fully_target_count = 0
top5_partly_target_count = 0
top5_no_target_count = 0

item = None
for tmp_item in longmemeval_data:
    if tmp_item['question_type'] in question_type and "_abs" not in tmp_item['question_id']:
        item = tmp_item
        total_count += 1
        print(f"Order: {total_count}. Initializing Conversation for {tmp_item['question_id']}...")
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
        current_question = item["question"]
        current_answer = item["answer"]
        question_date = item["question_date"]
        evidence_session_ids = item["answer_session_ids"]
        print("Emb computation...")
        retriever.compute_emb_for_conversation(tmp_conversation, strategy='session_facts', server='openai')
        retriever.compute_scores_for_conversation(current_question, tmp_conversation, server='openai')
        top_k_facts, top_k_scores, top_k_sids = retriever.get_top_k(tmp_conversation,)

        for i in range(len(top_k_scores)):
            print(f"{top_k_scores[i]}\t{top_k_sids[i]}\t{top_k_facts[i]}")
        target_result = get_target(evidence_session_ids, top_k_sids)
        if target_result == "fully-target":
            fully_target_count += 1
        elif target_result == "partly-target":
            partly_target_count += 1
        elif target_result == "no-target":
            no_target_count += 1
        
        print(item['question_type'])
        print(f"[Top10] Target Result of Order {total_count} is: {target_result}.")
        print(f"[Top10] Right Now: \nTotal: {total_count} \nFully Target: {fully_target_count} \nPartly Target: {partly_target_count} \nNo Target: {no_target_count}")

        top5_target_result = get_target(evidence_session_ids, top_k_sids[0:5])
        if top5_target_result == "fully-target":
            top5_fully_target_count += 1
        elif top5_target_result == "partly-target":
            top5_partly_target_count += 1
        elif top5_target_result == "no-target":
            top5_no_target_count += 1
        
        print(f"[Top5] Target Result of Order {total_count} is: {top5_target_result}.")
        print(f"[Top5] Right Now: \nTotal: {total_count} \nFully Target: {top5_fully_target_count} \nPartly Target: {top5_partly_target_count} \nNo Target: {top5_no_target_count}")


print(f"[Top10] The Final Retrieval Target Results: \nTotal: {total_count} \nFully Target: {fully_target_count} \nPartly Target: {partly_target_count} \nNo Target: {no_target_count}")
print(f"[Top5] The Final Retrieval Target Results: \nTotal: {total_count} \nFully Target: {top5_fully_target_count} \nPartly Target: {top5_partly_target_count} \nNo Target: {top5_no_target_count}")
