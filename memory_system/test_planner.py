from planner.planner import Planner
from llms.packaged_llms import UnifiedLLM, get_messages
from utils.dates import date2datetime
import os
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

data_path = "../data/longmemeval_oracle"
with open(data_path, "r") as f:
    longmemeval_data = json.load(f)
print(f"Dataset(path: {data_path}) is loaded. Total number of samples: {len(longmemeval_data)}.")

LLM = UnifiedLLM("Qwen2.5-3B-Instruct")
my_planner = Planner(LLM)



# Test Time Range Inference
for item in longmemeval_data:
    if item['question_type'] not in ['single-session-assistant']:
        continue
    question = item['question']
    question_date = str(item['question_date'])
    question_type = item['question_type']
    inferred_date = my_planner.get_time_range(question, question_date)

    print("-"*100)
    print(f"Question Type: {question_type}\nQuestion: {question}\nDate: {question_date}\nInferred Time Range: {inferred_date}")
# 感觉幻觉问题太严重了..



# Test Question Classification
# count = 0
# correct_count = 0
# question_type = ["single-session-user","multi-session","temporal-reasoning","knowledge-update"]

# for item in longmemeval_data:
#     if item['question_type'] in question_type:
#         count += 1
#         test_question = item['question']
#         result = my_planner.get_question_type(test_question)
#         print(f"Question: {test_question}\nClassification: {result}\n\n")
#         if result in "Fact" or result == "Fact":
#             correct_count += 1

# print(f"\n\nThe classification accuracy for question type {question_type} is {correct_count}/{count} = {correct_count/count * 100:.2f}%.")

