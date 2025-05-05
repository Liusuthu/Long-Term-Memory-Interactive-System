from llms.packaged_llms import UnifiedLLM, get_messages
from llms.packaged_llms import normal_system_prompt


judge_model_name = "Qwen2.5-3B-Instruct"


judge_system_prompt = "You are a supportive and impartial AI assistant acting as a judge. Your role is to help the user determine whether a given answer to a question is correct, providing clear and objective reasoning when necessary."
judge_instruction = "You are a professional and objective evaluator. You will be given a question, a correct answer, and a candidate answer to be judged. Based on the content of the question and the correct answer, analyze whether the candidate answer is correct. \nYour output should include a judgement (Correct, Incorrect, or Partially Correct) and a brief explanation. \nSome notifications: \n{}\n\nQuestion: {}\nCorrect Answer: {} \n\nCandidate Answer: {}\n\nYour output: (Provide the judgment and a brief explanation on two separate lines; do not generate any other content)"

notifications = """
1) If the candidate answer is equivalent to the correct answer or contains the intermediate steps to get the correct answer, you should also regard it as correct. 
2) If the candidate answer only contains a subset of the information required by the correct answer, you should also regard it as incorrect.
3) If the candidate answer contains some previous information along with an updated answer, the candidate answer should be considered as correct as long as the updated answer is the required correct answer
4) For user preference related questions, if the candidate answer satisfies the desired correct answer, regard it as correct, otherwise incorrect. The candidate answer does not need to reflect all the points in the rubric, it is regarded as correct as long as it recalls and utilizes the user's personal information correctly.
"""

class LLMJudge:
    def __init__(self, llm_judge:UnifiedLLM,):
        self.model_name = judge_model_name
        self.judge_model = llm_judge

    def judge(self, question, expected_answer, candidate_answer):
        messages = get_messages(
            system_prompt=judge_system_prompt,
            user_prompt=judge_instruction.format(notifications, question, expected_answer, candidate_answer)
        )
        return self.judge_model.generate(messages)