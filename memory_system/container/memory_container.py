import types

class Round:
    def __init__(self, message:list=None, extract_keys:bool=False, llm_extractor=None):
        """
        A round between a user and an assistant.
        Args:
            message (list): A list of two dictionaries, one for the user and one for the assistant.
                            Each dictionary must contain 'role' and 'content'.
            extract_keys (bool): If True, extract a key from the assistant's message using llm_extractor.
            llm_extractor: An LLM to act as a key-extractor
        """

        self.round_keys = []
        self.round_facts = []
        self.user = None
        self.assistant = None

        if not isinstance(message, list) or len(message) != 2:
            raise ValueError("messages must be a list of two dictionaries: one user and one assistant message.")
        user_msg = message[0]["content"]
        assistant_msg = message[1]["content"]
        if user_msg is None or assistant_msg is None:
            raise ValueError("messages must contain one 'user' and one 'assistant' entry.")
        
        self.user = user_msg
        self.assistant = assistant_msg
        if extract_keys:
            if llm_extractor is None:
                raise ValueError("llm_extractor must be provided if extract_keys is True.")
            self.round_keys = self.extract_keys(llm_extractor)



    def __repr__(self):
        return f"Round(\n  user={self.user[:20]!r}..., \n  assistant={self.assistant[:20]!r}...,\n  round_keys={self.round_keys},\n  round_facts={self.round_facts} \n)"

    def show_round(self):
        print(f"User: {self.user}")
        print(f"Assistant: {self.assistant}")


    def extract_keys(self, llm_extractor):
        ...



class Session:
    def __init__(self, session_id:str=None, session_time:str=None, session:list=None, extract_keys:bool=False, llm_extractor=None):
        """
        A session between a user and an assistant. Which contains multiple rounds.
        Args:
            session_id (str): The ID of the session.
            session_time (str): The chat time of this session.
            session (list): A list of rounds.
            extract_keys (bool): If True, extract keys from the assistant's messages using llm_extractor.
            llm_extractor: An LLM to act as a key-extractor
        """
        self.session_id = session_id
        self.session_time = session_time
        self.session_keys = []
        self.session_facts = []
        self.rounds = []

        self.num_rounds = len(session)//2
        for idx in range(self.num_rounds):
            user_msg = session[2*idx]["content"]
            assistant_msg = session[2*idx+1]["content"]
            if user_msg is None or assistant_msg is None:
                raise ValueError("messages must contain one 'user' and one 'assistant' entry.")
            # TODO:下面的format也可以，但感觉不太优雅..
            round = Round([{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_msg}])
            self.rounds.append(round)

        if extract_keys:
            if llm_extractor is None:
                raise ValueError("llm_extractor must be provided if extract_keys is True.")
            self.session_keys = self.extract_keys(llm_extractor)
        

    def __repr__(self):
        # TODO:打印格式需要优化得可读性更好一些
        return f"Session(\n  session_id={self.session_id!r}, \n  session_time={self.session_time!r}, \n  num_rounds={self.num_rounds}, \n  session_keys={self.session_keys}, \n  session_facts={self.session_facts} \n)"
    
    def show_session(self):
        print("-"*60)
        for i in range(self.num_rounds):
            print(f"Round {i+1}:")
            self.rounds[i].show_round()
            print("-"*60)


    def extract_keys(self, llm_extractor):
        ...




class Conversation:
    ...




msg=[
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm fine, thank you! How can I assist you today?"}
]
round = Round(msg)
print(round)
round.show_round()


sess=[
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm fine, thank you! How can I assist you today?"},
    {"role": "user", "content": "What is the weather like today?"},
    {"role": "assistant", "content": "The weather is sunny with a high of 25 degrees."},
    {"role": "user", "content": "Can you tell me a joke?"},
    {"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side!"},
]
session = Session(session=sess)
print(session)
session.show_session()