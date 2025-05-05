from llms.open_sourced import load_llm_model, load_qwen3_model, llm_generate, qwen3_generate
from llms.close_sourced import get_client, get_response
from llms.packaged_llms import UnifiedLLM, get_messages
from utils.chunks import session2context

# export PYTHONPATH=$PYTHONPATH:/home/limusheng/Long-Term-Memory-Interactive-System

normal_system_prompt = "You are a supportive AI assistant, focused on helping the user accomplish their tasks by following instructions carefully and precisely."

plain_reader_prompt = "Below is a transcript of a conversation between a human user and an AI assistant, which consists of multiple sessions in temporal oder between the human user and the AI assistant. The timestamps are also provided. This conversation contains some key information about a question. You should answer the question after carefully analyzing the long conversation. The raw text of the conversation and the question are below:\n\nConversation:\n\n{}\n\nQuestion: {}\n\nYour Answer: (answer according to the conversation context; be precise and stick to the facts. Do not generate anything else.)"

class PlainReader():
    "Simply use the retrieved text as input"
    def __init__(self, reader_llm:UnifiedLLM):
        self.reader_model = reader_llm
    

    def get_answer(self, retrieval_type, retrieval_list, question, question_date=None):
        """
        The function used to generate answer from retrieved chunks.
        Args:
            retrieval_type (str): Choose from `round`, `session` or `hybrid`.
            retrieval_list (list): Retrieved chunks, organized in a list.
            question_date (str): The date when the question is asked, very important for temporal reasoning.
        """
        if retrieval_type == "round":
            ...
            # TODO: 根据具体呈现的retrieval_list形式来做设计，后面都是

        elif retrieval_type == "session":
            context = session2context(retrieval_list)
            if question_date is not None:
                context = f"Question Date: {question_date}\n\n" + context
            messages = get_messages(normal_system_prompt, plain_reader_prompt.format(context, question))
            result = self.reader_model.generate(messages)
            return result
        elif retrieval_type == "hybrid":
            ...

        else:
            raise ValueError(f"Argument retrieval_type should be one of `round`, `session` or `hybrid`, but now is {retrieval_type}.")
        



class CoNReader():
    "First filter the retrieved chunks, then get answer."
    def __init__(self, reader_llm:UnifiedLLM):
        self.reader_model = reader_llm
        



    def get_answer(self, retrieval_type, retrieval_list):
        """
        The function used to generate answer from retrieved chunks.
        Args:
            retrieval_type (str): Choose from `round`, `session` or `hybrid`.
            retrieval_list (list): Retrieved chunks, organized in a list.
        """
        if retrieval_type == "round":
            ...

        elif retrieval_type == "session":
            ...
        
        elif retrieval_type == "hybrid":
            ...

        else:
            raise ValueError(f"Argument retrieval_type should be one of `round`, `session` or `hybrid`, but now is {retrieval_type}.")


