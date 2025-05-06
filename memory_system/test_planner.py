from planner.planner import Planner
from llms.packaged_llms import UnifiedLLM, get_messages
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("-"*40 + " Stage 1: Load Data " +"-"*40)
data_path = "../data/longmemeval_oracle"
with open(data_path, "r") as f:
    longmemeval_data = json.load(f)
print(f"Dataset(path: {data_path}) is loaded. Total number of samples: {len(longmemeval_data)}.")

LLM = UnifiedLLM("Qwen2.5-3B-Instruct")
my_planner = Planner(LLM)

count = 0
correct_count = 0
question_type = "multi-session"

for item in longmemeval_data:
    if item['question_type'] == question_type:
        count += 1
        test_question = item['question']
        result = my_planner.get_question_type(test_question)
        print(f"Question: {test_question}\nClassification: {result}\n\n")
        if result in "Fact" or result == "Fact":
            correct_count += 1

print(f"\n\nThe classification accuracy for question type {question_type} is {correct_count}/{count} = {correct_count/count * 100:.2f}%.")

