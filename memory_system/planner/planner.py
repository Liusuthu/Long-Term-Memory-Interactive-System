"When confronted with a question, generate a list of steps to find the answer."

from llms.packaged_llms import UnifiedLLM, get_messages

class Planner:
    def __init__(self, core_model_name:str="Qwen2.5-14B-Instruct"):
        print(f"Loading Planner LLM Core: {core_model_name}...")
        self.llm = UnifiedLLM(core_model_name)

    def generate_plan(self, question,):
        ...