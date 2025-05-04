from llms.packaged_llms import UnifiedLLM, get_messages
from llms.packaged_llms import normal_system_prompt


judge_model_name = "Qwen2.5-3B-Instruct"


judge_system_prompt = "You are a supportive and impartial AI assistant acting as a judge. Your role is to help the user determine whether a given answer to a question is correct, providing clear and objective reasoning when necessary."
judge_instruction = "You are a professional and objective evaluator. You will be given a question, an expected answer, and a candidate answer to be judged. Based on the content of the question and the expected answer, analyze whether the candidate answer is correct. \nYour output should include a judgment (Correct, Incorrect, or Partially Correct) and a brief explanation.\n\nQuestion: {}\nExpected Answer: {} \n\nCandidate Answer: {}\n\nYour output: (Provide the judgment and a brief explanation on two separate lines; do not generate any other content)"


class LLMJudge:
    def __init__(self, llm_judge:UnifiedLLM,):
        self.model_name = judge_model_name
        self.judge_model = llm_judge

    def judge(self, question, expected_answer, candidate_answer):
        messages = get_messages(
            system_prompt=judge_system_prompt,
            user_prompt=judge_instruction.format(question, expected_answer, candidate_answer)
        )
        return self.judge_model.generate(messages)