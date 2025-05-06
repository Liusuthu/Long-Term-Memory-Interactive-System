"When confronted with a question, generate a list of steps to find the answer."

from llms.packaged_llms import UnifiedLLM, get_messages
from container.memory_container import normal_system_prompt

# TODO: 需要给一个action的约束集合，这个集合最好是通过想象+分析题目得到。

classify_question_prompt = """
In a long-term User-Assistant conversation background, you will be provided with a question, your task is to classify this question into one of three types: Fact, Preference, Assistant. Their definitions are below:

Fact: Questions that require retrieving factual events from past conversations and answering based on those facts, those facts are usually about specific person, life events, personal experience, numbers, locations, or dates.
Preference: It usually involves the user seeking advice from the assistant, requiring the assistant to provide suggestions, ideas, probable reasons, or offer some reflections on the user's assumptions.
Assistant-Related: The user and the assistant have previously had a conversation or chat on a certain topic, and the user asks the assistant to remind the user something about it, and to reintroduce some of the previous statements.

-----------------------------
Some examples of classification:
Fact: 
- Where did I redeem a $5 coupon on coffee creamer?
- How long is my daily commute to work?
- What was the first issue I had with my new car after its first service?
- How many bikes do I currently own?
- What type of rice is my favorite?

Preference:
- Can you recommend some resources where I can learn more about video editing?
- I've been sneezing quite a bit lately. Do you think it might be my living room?
- What should I serve for dinner this weekend with my homegrown ingredients?
- I'm trying to decide whether to buy a NAS device now or wait. What do you think?
- I noticed my bike seems to be performing even better during my Sunday group rides. Could there be a reason for this?

Assistant-Related:
- I remember you told me about the refining processes at CITGO's three refineries earlier. Can you remind me what kind of processes are used at the Lake Charles Refinery?
- I'm checking our previous chat about the shift rotation sheet for GM social media agents. Can you remind me what was the rotation for Admon on a Sunday?
- I'm planning to revisit Orlando. I was wondering if you could remind me of that unique dessert shop with the giant milkshakes we talked about last time?
- I was looking back at our previous conversation about Native American powwows and I was wondering, which traditional game did you say was often performed by skilled dancers at powwows?
- I'm looking back at our previous chat and I wanted to confirm, how many times did the Chiefs play the Jaguars at Arrowhead Stadium?
-----------------------------

Now it's your term to do a classification:
Question: {}
Type: (According to the information above, classify into one of Fact, Preference and Assistant-Related; do not generate anything else)
"""


time_range_prompt="""
You will be given a question from a human user asking about some prvious events, as well as the time the question is asked. Infer a potential time range such that the events happening in this range is likely to help to answer the question (a start date and an end date). Write a dict with two fields: ”start” and ”end” representing start and end date. Write the date in the form YYYY/MM/DD. If the question does not have any temporal reference, do not attempt to guess a time range, just say N/A. You can allow some flexible expansion in your inferred time range.

Here are some examples: 
{}

Question: {}
Question Time: {}

Inferred Time Range: (don't generate anything else except the dict and N/A)
"""

time_range_examples = """
Question: What kitchen appliance did I buy 10 days ago?
Question Time: 2023/03/25 (Sat) 18:04
Inferred Time Range: {"start": "2023/03/14", "end": "2023/03/25"}

Question: What did I do with Rachel on the Wednesday two months ago?
Question Time: 2023/04/01 (Sat) 20:22
Inferred Time Range: {"start": "2023/01/31", "end": "2023/02/06"}

Question: I mentioned cooking something for my friend a couple of days ago. What was it?
Question Time: 2022/04/12 (Tue) 13:26
Inferred Time Range: {"start": "2022/04/09", "end": "2022/04/12"}

Question: Which pair of shoes did I clean last month?
Question Time: 2023/05/30 (Tue) 01:50
Inferred Time Range: {"start": "2023/03/30", "end": "2023/05/30"}

Question: What was the social media activity I participated 5 days ago?
Question Time: 2023/03/20 (Mon) 05:50
Inferred Time Range: {"start": "2023/03/15", "end": "2023/03/20"}

Question: Where did I attend the religious activity last week?
Question Time: 2023/04/10 (Mon) 08:05
Inferred Time Range: {"start": "2023/03/31", "end": "2023/04/10"}

Question: How many weeks ago did I start using the cashback app 'Ibotta'?
Question Time: 2023/05/06 (Sat) 05:25
Inferred Time Range: {"start": "2023/03/18", "end": "2023/05/06"}
"""


class Planner:
    "An LLM Agent That Helps Generate The Problem-Solution Trajectory."
    def __init__(self, core_model:UnifiedLLM):
        self.llm = core_model

    def get_question_type(self, question,):
        messages = get_messages(normal_system_prompt, classify_question_prompt.format(question))
        result = self.llm.generate(messages)
        return result
    
    def get_time_range(self, question, question_date):
        "The prompt is adapted from LongMemEval Query Expansion."
        messages = get_messages(normal_system_prompt, time_range_prompt.format(time_range_examples, question, question_date))
        result = self.llm.generate(messages)
        return result
        # 观察到time range可能并不是特别准确或者可能有偏差，可以提取后人工手动前后添加几天/月