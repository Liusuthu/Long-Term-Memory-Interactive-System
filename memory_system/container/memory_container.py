"The memory storage structure of our memory system"

import types
import ast
from termcolor import colored
from llms.packaged_llms import UnifiedLLM
from llms.packaged_llms import get_messages

# test_llm_extractor = UnifiedLLM("Qwen2.5-14B-Instruct")


# 目前Prompt先基于LongMemEval论文中的做法简单修改实现，以跑通流程为主
normal_system_prompt = "You are a supportive AI assistant, focused on helping the user accomplish their tasks by following instructions carefully and precisely."

extract_round_keys_prompt = "Below is a transcript of a round of conversation between a human user and an AI assistant. Generate a list of keyphrases that best represent this round of conversation. These keyphrases don't necessarily have to originally from conversation. The generated keyphrases should be meaningful, such as key nouns, adjectives, specific values, and corresponding entities (but not limited to those mentioned), other than just simple words. Separate each keyphrase with a semicolon. If there are no significant keyphrases in this round of conversation, simply generate 'N/A'. \n\nDialogue content:\n{}. \n\nA list of keyphrases: (a list of keyphrases in str format; do not generate anything else. Better to have nothing than to have something of poor quality.)"
extract_round_summary_key_prompt = "Below is a transcript of a round of conversation between a human user and an AI assistant. Generate a phrase that best summarize the content of this round of conversation(not anything like resquest and response!). This phrase doesn't necessarily have to be originally from conversation. \n\nDialogue content:\n{} \n\nPhrase: (a phrase that best summarizes this round of conversation in string format; do not generate anything else.)"
extract_round_facts_prompt = "You will be given a round of conversation from a human user to an AI assistant. Extract all the personal information, life events, experience, and preferences related to the user and the assistant. Make sure you include all details such as life events, personal experience, preferences, specific numbers, locations, or dates. State each piece of information in a simple sentence. Put these sentences in a json list, each element being a standalone personal fact about the user or the assistant. Minimize the coreference across the facts, e.g., replace pronouns with actual entities. If there is no specific events, personal information, or preference mentioned, just generate an empty list. \n\nDialogue content:\n{}. \n\nPersonal facts about the user (a list of strings in json format; do not generate anything else):"

extract_session_keys_prompt = "Below is a transcript of a chat session between a human user and an AI assistant, which consists of multiple rounds between the human user and the AI assistant. Generate a list of keyphrases that best represent this session of conversation. These keyphrases don't necessarily have to originally from conversation. The generated keyphrases should be meaningful, such as key nouns, adjectives, specific values, and corresponding entities (but not limited to those mentioned), other than just simple words. Separate each keyphrase with a semicolon. If there are no significant keyphrases in this session of conversation, simply generate 'N/A'. \n\nDialogue content:\n{}. \n\nA list of keyphrases: (a list of keyphrases in str format; do not generate anything else. Better to have nothing than to have something of poor quality.)"
extract_session_summary_key_prompt = "Below is a transcript of a session of chat between a human user and an AI assistant. Generate a phrase that best summarize the content of this session of conversation(not anything like resquest and response!). This phrase doesn't necessarily have to be originally from conversation. \n\nDialogue content:\n{} \n\nPhrase: (a phrase that best summarizes this session of conversation in string format; do not generate anything else.)"
extract_session_facts_prompt = "You will be given a session of conversation between a human user and an AI assistant. Extract all the personal information, life events, experience, and preferences related to the user. Also conclude the key facts or information said by the assistant(only the most important sentences). Make sure you include all details such as life events, personal experience, preferences, specific numbers, locations, topics or dates. State each piece of information in a simple sentence. Put these sentences in a json list, each element being a standalone personal fact about the user or the assistant. Minimize the coreference across the facts, e.g., replace pronouns with actual entities. If there is no specific events, personal information, or preference mentioned, just generate an empty list. \n\nDialogue content:\n{}. \n\nFacts about the user and the assistant (a list of strings in json format; do not generate anything else):"

extract_conversation_keys_prompt = "Below is a transcript of a conversation between a human user and an AI assistant, which consists of multiple sessions between the human user and the AI assistant. Generate a list of keyphrases that best represent this conversation. These keyphrases don't necessarily have to originally from the conversation. The generated keyphrases should be meaningful, such as key nouns, adjectives, specific values, and corresponding entities (but not limited to those mentioned), other than just simple words. Separate each keyphrase with a semicolon. If there are no significant keyphrases in this conversation, simply generate 'N/A'. \n\nConversation:\n{}. \n\nA list of keyphrases: (a list of keyphrases in str format; do not generate anything else. Better to have nothing than to have something of poor quality.)"
extract_conversation_summary_key_prompt = "Below is a transcript of a conversation between a human user and an AI assistant. Generate a phrase that best summarize the content of this conversation(not anything like resquest and response!). This phrase doesn't necessarily have to be originally from conversation. \n\nConversation  content:\n{} \n\nPhrase: (a phrase that best summarizes this conversation in string format; do not generate anything else.)"
extract_conversation_facts_prompt = "You will be given a conversation from a human user to an AI assistant. Extract all the personal information, life events, experience, and preferences related to the user and the assistant. Make sure you include all details such as life events, personal experience, preferences, specific numbers, locations, or dates. State each piece of information in a simple sentence. Put these sentences in a json list, each element being a standalone personal fact about the user or the assistant. Minimize the coreference across the facts, e.g., replace pronouns with actual entities. If there is no specific events, personal information, or preference mentioned, just generate an empty list. \n\nConversation content:\n{}. \n\nPersonal facts about the user (a list of strings in json format; do not generate anything else):"


class Round:
    "A round between a user and an assistant."
    def __init__(self, message:list=None, extract_keys:bool=False, extract_facts:bool=False, llm_extractor:UnifiedLLM=None):
        """
        Args:
            message (list): A list of two dictionaries, one for the user and one for the assistant.
                            Each dictionary must contain 'role' and 'content'.
            extract_keys (bool): If True, extract a key from this round using llm_extractor.
            llm_extractor: A pre-packaged and ready-to-use LLM to act as a key-extractor.
        """

        self.round_summary_key = None
        self.has_round_summary_key = False
        self.round_keys = []
        self.has_round_keys = False
        self.round_facts = []
        self.has_round_facts = False
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
            self.extract_round_keys(llm_extractor)
        if extract_facts:
            if llm_extractor is None:
                raise ValueError("llm_extractor must be provided if extract_facts is True.")
            self.extract_round_facts(llm_extractor)


    def __repr__(self):
        return f"Round(\n  user={self.user[:20]!r}..., \n  assistant={self.assistant[:20]!r}...,\n  round_summary_key={self.round_summary_key}, \n  round_keys={self.round_keys},\n  round_facts={self.round_facts} \n)"


    def show_round(self):
        print(colored("User:",color="cyan",attrs=["bold"]),f"{self.user}")
        print(colored("Assistant:",color="cyan",attrs=["bold"]),f"{self.assistant}")

    def get_round_str(self):
        return f"USER: {self.user}\nASSISTANT: {self.assistant}"


    def extract_round_keys(self, llm_extractor):
        # For normal keys
        messages = get_messages(normal_system_prompt, extract_round_keys_prompt.format(self.get_round_str()))
        result = llm_extractor.generate(messages)
        if result != 'N/A':
            self.has_round_keys = True
            keyword_list = result.replace("\n", "").replace("\t", "").split("; ")
            self.round_keys = keyword_list
        # For summary key
        messages = get_messages(normal_system_prompt, extract_round_summary_key_prompt.format(self.get_round_str()))
        result = llm_extractor.generate(messages)
        self.has_round_summary_key = True
        self.round_summary_key = result


    def extract_round_facts(self, llm_extractor):
        messages = get_messages(normal_system_prompt, extract_round_facts_prompt.format(self.get_round_str()))
        result = llm_extractor.generate(messages)
        result_list = ast.literal_eval(result)
        if len(result_list) != 0:
            self.has_round_facts = True
            self.round_facts = result_list




class Session:
    "A session between a user and an assistant. Which contains multiple rounds."
    def __init__(self, session:list, session_id:str=None, session_date:str=None, extract_keys:bool=False, extract_facts:bool=False, llm_extractor:UnifiedLLM=None):
        """
        Args:
            session_id (str): The ID of the session.
            session_date (str): The chat time of this session.
            session (list): A list of rounds.
            extract_keys (bool): If True, extract keys from this session using llm_extractor.
            llm_extractor: An LLM to act as a key-extractor.
        """
        self.session_id = session_id
        self.session_date = session_date
        self.session_summary_key = None
        self.has_session_summary_key = False
        self.session_keys = []
        self.has_session_keys = False
        self.session_facts = []
        self.has_session_facts = False
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
            self.extract_session_keys(llm_extractor)
        if extract_facts:
            if llm_extractor is None:
                raise ValueError("llm_extractor must be provided if extract_facts is True.")
            self.extract_session_facts(llm_extractor)


    def __repr__(self):
        return f"Session(\n  session_id={self.session_id!r}, \n  session_date={self.session_date!r}, \n  num_rounds={self.num_rounds}, \n  session_summary_key={self.session_summary_key}, \n  session_keys={self.session_keys}, \n  session_facts={self.session_facts} \n)"
    
    def show_session(self):
        print(colored("Session ID:","green",attrs=["bold"]),self.session_id)
        print(colored("Session date:","green",attrs=["bold"]),self.session_date)
        print("-"*60)
        for i in range(self.num_rounds):
            print(colored(f"Round {i+1}:",color="cyan",attrs=["bold"]))
            self.rounds[i].show_round()
            print("-"*60)

    def get_session_str(self, str_type="brief"):
        """
        Args:
            str_type (str): The returned session string type, choose from `brief` and `detailed`.
        """
        session_str = ""
        if str_type == 'brief':
            for i in range(self.num_rounds):
                session_str += f"USER: {self.rounds[i].user}\nASSISTANT: {self.rounds[i].assistant}\n"

        elif str_type == 'detailed':
            session_str += "-"*60 + '\n'
            for i in range(self.num_rounds):
                session_str += f"Round {i+1}:\n"
                session_str += f"USER: {self.rounds[i].user}\nASSISTANT: {self.rounds[i].assistant}\n"
                session_str += "-"*60 + '\n'
        else:
            raise ValueError(f"Get session type {str_type} is not supported, choose from `brief` and `detailed`.")
        return session_str

        


    def extract_session_keys(self, llm_extractor):
        # For normal keys
        messages = get_messages(normal_system_prompt, extract_session_keys_prompt.format(self.get_session_str()))
        result = llm_extractor.generate(messages)
        if result != 'N/A':
            self.has_session_keys = True
            keyword_list = result.replace("\n", "").replace("\t", "").split("; ")
            self.session_keys = keyword_list
        # For summary key
        messages = get_messages(normal_system_prompt, extract_session_summary_key_prompt.format(self.get_session_str()))
        result = llm_extractor.generate(messages)
        self.has_session_summary_key = True
        self.session_summary_key = result


    def extract_session_facts(self, llm_extractor, max_retries=3):
        messages = get_messages(normal_system_prompt, extract_session_facts_prompt.format(self.get_session_str()))
        retries = 0
        while retries < max_retries:
            try:
                result = llm_extractor.generate(messages)
                result_list = ast.literal_eval(result)
                if len(result_list) != 0:
                    self.has_session_facts = True
                    self.session_facts = result_list
                    return  # 成功后退出
                else:
                    return  # 结果为空，也直接返回
            except (ValueError, SyntaxError) as e:
                retries += 1
                if retries >= max_retries:
                    print(f"Exceeded {max_retries} retries, skipping this session.")
                    return  # 超过最大重试次数后跳过当前 session（坏session，不要也罢）




class Conversation:
    "A full conversation between a user and an assistant, which is associate with a QA."
    def __init__(self, sessions:list, session_id_list:list, session_date_list:str, date:str=None, id:str=None, extract_keys:bool=False, extract_facts:bool=False, llm_extractor:UnifiedLLM=None):
        """
        Args:
            session (list): A list of lists(session), annotated with session_id and session_date.
            date (str): The date of this conversation.
            id (str): Conversation ID.
            extract_keys (bool): If True, extract a key from this conversation using llm_extractor.
            llm_extractor: An LLM to act as a key-extractor.
        """
        self.conversation_date = date
        self.conversation_id = id
        self.num_sessions = len(sessions)
        self.sessions = []
        self.conversation_summary_key = None
        self.has_conversation_summary_key = False
        self.conversation_keys = []
        self.has_conversation_keys = False
        self.conversation_facts = []
        self.has_conversation_facts = False

        for idx, session in enumerate(sessions):
            cur_session = Session(session=session, session_id=session_id_list[idx], session_date=session_date_list[idx])
            self.sessions.append(cur_session)

        if extract_keys:
            if llm_extractor is None:
                raise ValueError("llm_extractor must be provided if extract_keys is True.")
            self.extract_conversation_keys(llm_extractor)
        if extract_facts:
            if llm_extractor is None:
                raise ValueError("llm_extractor must be provided if extract_facts is True.")
            self.extract_conversation_facts(llm_extractor)


    def __repr__(self):
        return f"Conversation(\n  conversation_id={self.conversation_id!r}, \n  conversationn_date={self.conversation_date!r}, \n  num_sessions={self.num_sessions}, \n  conversation_summary_key={self.conversation_summary_key}, \n  conversation_keys={self.conversation_keys}, \n  session_facts={self.conversation_facts} \n)"

    def show_conversation(self):
        print(colored("Conversation ID:",color="yellow",attrs=["bold"]),self.conversation_id)
        print(colored("Conversation date:",color="yellow",attrs=["bold"]), self.conversation_date)
        print("-"*60)
        for i in range(self.num_sessions):
            print(colored(f"Session {i}:",color="green",attrs=["bold"]))
            self.sessions[i].show_session()

    def get_conversation_str(self):
        conversation_str = ""
        for i in range(self.num_sessions):
            conversation_str += f"Session {i}:/n"
            cur_session = self.sessions[i]
            conversation_str += cur_session.get_session_str()
            conversation_str += "-"*60 + '\n'
        return conversation_str




    def extract_conversation_keys(self, llm_extractor):
        #  For normal keys
        messages = get_messages(normal_system_prompt, extract_conversation_keys_prompt.format(self.get_conversation_str()))
        result = llm_extractor.generate(messages)
        if result != 'N/A':
            self.has_conversation_keys = True
            keyword_list = result.replace("\n", "").replace("\t", "").split("; ")
            self.conversation_keys = keyword_list
        # For summary key
        messages = get_messages(normal_system_prompt, extract_conversation_summary_key_prompt.format(self.get_conversation_str()))
        result = llm_extractor.generate(messages)
        self.has_conversation_summary_key = True
        self.conversation_summary_key = result



    def extract_conversation_facts(self, llm_extractor):
        messages = get_messages(normal_system_prompt, extract_conversation_facts_prompt.format(self.get_conversation_str()))
        result = llm_extractor.generate(messages)
        result_list = ast.literal_eval(result)
        if len(result_list) != 0:
            self.has_conversation_facts = True
            self.conversation_facts = result_list



msg=[
    # {"role": "user", "content": "Hello, how are you?"},
    # {"role": "assistant", "content": "I'm fine, thank you! How can I assist you today?"},
    # {"role": "user", "content": "What is the weather like today?"},
    # {"role": "assistant", "content": "The weather is sunny with a high of 25 degrees."}
    # {"role": "user", "content": "Can you tell me a joke?"},
    # {"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side!"},
    {"role":"user", "content":"I'm thinking of getting my car detailed soon. Do you know any good detailers in the area or have any recommendations? By the way, I just got my car serviced for the first time on March 15th, and it was a great experience."},
    {"role":"assistant", "content": "I'm happy to help you with your question! However, I'm a large language model, I don't have personal experiences or knowledge of specific locations or businesses. I can provide some general advice and suggestions on how to find a good detailer in your area.\n\nTo find a reputable and reliable detailer, you can try the following:\n\n1. Online Reviews: Check review websites such as Yelp, Google Reviews, or Facebook Reviews to see what other customers have to say about their experiences with local detailers.\n2. Ask for Referrals: Ask friends, family, or coworkers who have had their cars detailed recently for recommendations.\n3. Check Certifications: Look for detailers who are certified by organizations such as the International Detailing Association (IDA) or the National Institute of Automotive Service Excellence (ASE).\n4. Check Their Work: Ask to see before-and-after photos of their work or visit their shop to see the quality of their work firsthand.\n5. Get Quotes: Get quotes from several detailers and compare prices and services offered.\n\nAs for your car service experience on March 15th, I'm glad to hear it was a great experience! It's always important to keep your car well-maintained, and it sounds like you're taking good care of your vehicle.\n\nRemember to always do your research and choose a reputable detailer to ensure your car receives the best care possible. Good luck in your search!"},
    # {"role":"user","content":"I'll definitely ask those questions when I visit the detailer. And yeah, the GPS issue was a bit frustrating, but at least it was resolved quickly. By the way, do you know if there are any specific car wax products that are better suited for cars with advanced paint finishes, like my silver Honda Civic?"},
    # {"role":"assistant","content":"Advanced paint finishes, like those found on many modern vehicles, require special care to maintain their appearance and protect them from damage. When it comes to car wax, you'll want to look for products that are specifically designed for these types of finishes. Here are some tips to help you choose the right wax for your silver Honda Civic:\n\n1. **Look for waxes labeled as \"safe for advanced paint finishes\" or \"clear coat safe\"**: These waxes are formulated to work with the unique properties of advanced paint finishes, which can be more sensitive to certain chemicals.\n2. **Opt for a wax with a high Carnauba content**: Carnauba wax is a natural wax that provides an excellent barrier against the elements and is gentle on paint. Look for waxes with a high percentage of Carnauba (usually 20% or higher).\n3. **Consider a synthetic sealant or ceramic coating**: These products offer longer-lasting protection than traditional waxes and can provide an additional layer of defense against the elements. They're especially useful for cars with advanced paint finishes.\n4. **Avoid waxes with abrasive ingredients**: Some waxes contain abrasives that can scratch or mar advanced paint finishes. Look for waxes that are labeled as \"abrasive-free\" or \"non-abrasive.\"\n5. **Check the pH level**: A pH level close to neutral (around 7) is ideal for most paint finishes. Avoid waxes with high or low pH levels, as they can potentially damage the paint.\n\nSome popular car wax products that are suitable for advanced paint finishes include:\n\n* Meguiar's Ultimate Liquid Wax\n* Chemical Guys Car Wax\n* Autogeek Garage Line Car Wax\n* P21S Concours Carnauba Wax\n* Gtechniq Crystal Serum Ceramic Coating (for a longer-lasting, more protective option)\n\nRemember to always read the instructions and follow the manufacturer's recommendations for application and removal.\n\nWhen you visit the detailer, ask them about the type of wax they use and whether it's suitable for your car's advanced paint finish. A good detailer will be knowledgeable about the products they use and can recommend the best option for your vehicle."}
]
# round = Round(msg)
# round.show_round()

# round.extract_round_facts(test_llm_extractor)
# round.extract_round_keys(test_llm_extractor)
# print(round)



# sess=[
#     {"role": "user", "content": "Hello, how are you?"},
#     {"role": "assistant", "content": "I'm fine, thank you! How can I assist you today?"},
#     {"role": "user", "content": "What is the weather like today?"},
#     {"role": "assistant", "content": "The weather is sunny with a high of 25 degrees."},
#     {"role": "user", "content": "Can you tell me a joke?"},
#     {"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side!"},
#     {"role": "user", "content": "Haha it's a cold joke! By the way, I'm going to go to the library to find the book Little Princess, do you know it?"},
#     {"role": "assistant", "content": "Of course! Do you want some information about it?"},
# ]

# sess = [
#     {
#         "role": "user",
#         "content": "I'm thinking of getting my car detailed soon. Do you know any good detailers in the area or have any recommendations? By the way, I just got my car serviced for the first time on March 15th, and it was a great experience.",
#         "has_answer": True
#     },
#     {
#         "role": "assistant",
#         "content": "I'm happy to help you with your question! However, I'm a large language model, I don't have personal experiences or knowledge of specific locations or businesses. I can provide some general advice and suggestions on how to find a good detailer in your area.\n\nTo find a reputable and reliable detailer, you can try the following:\n\n1. Online Reviews: Check review websites such as Yelp, Google Reviews, or Facebook Reviews to see what other customers have to say about their experiences with local detailers.\n2. Ask for Referrals: Ask friends, family, or coworkers who have had their cars detailed recently for recommendations.\n3. Check Certifications: Look for detailers who are certified by organizations such as the International Detailing Association (IDA) or the National Institute of Automotive Service Excellence (ASE).\n4. Check Their Work: Ask to see before-and-after photos of their work or visit their shop to see the quality of their work firsthand.\n5. Get Quotes: Get quotes from several detailers and compare prices and services offered.\n\nAs for your car service experience on March 15th, I'm glad to hear it was a great experience! It's always important to keep your car well-maintained, and it sounds like you're taking good care of your vehicle.\n\nRemember to always do your research and choose a reputable detailer to ensure your car receives the best care possible. Good luck in your search!",
#         "has_answer": False
#     },
#     {
#         "role": "user",
#         "content": "I'll definitely check out those review websites and ask around for referrals. Do you think it's a good idea to get a wax and detailing done every 3-4 months to maintain the car's appearance and protect the paint?",
#         "has_answer": False
#     },
#     {
#         "role": "assistant",
#         "content": "Regular waxing and detailing can definitely help maintain your car's appearance and protect the paint. The frequency of waxing and detailing depends on several factors, including:\n\n1. Climate: If you live in an area with harsh weather conditions, such as heavy rain, snow, or intense sunlight, you may need to wax and detail more frequently (every 2-3 months).\n2. Driving habits: If you drive frequently, park outdoors, or drive on rough roads, you may need to wax and detail more often to remove dirt, grime, and contaminants that can damage the paint.\n3. Paint type: Some paint finishes, like clear coat or ceramic coat, may require more frequent waxing and detailing to maintain their appearance and protection.\n\nIn general, waxing every 3-4 months is a good rule of thumb for most cars. This frequency can help:\n\n1. Protect the paint from UV rays, oxidation, and fading.\n2. Remove dirt, grime, and contaminants that can scratch or damage the paint.\n3. Maintain a glossy, showroom-like appearance.\n4. Make future washing and maintenance easier.\n\nHowever, if you want to take your car's appearance to the next level, you may consider a more frequent detailing schedule, such as:\n\n* Washing every 1-2 weeks to remove dirt and grime.\n* Waxing every 2-3 months to provide an extra layer of protection.\n* Detailing every 3-4 months to deep clean the interior and exterior.\n\nRemember to also consider your car's specific needs, such as any special paint finishes or requirements.\n\nIt's great that you're thinking about maintaining your car's appearance and protection. Regular waxing and detailing can definitely help extend the life of your car's paint and keep it looking its best!",
#         "has_answer": False
#     },
#     {
#         "role": "user",
#         "content": "Yup, that makes sense. I've been keeping track of my gas mileage too, and I've been getting around 32 miles per gallon, which is better than my old car.",
#         "has_answer": False
#     },
#     {
#         "role": "assistant",
#         "content": "That's great to hear! Keeping track of your gas mileage is a great way to monitor your car's performance and make adjustments to your driving habits or maintenance routine as needed.\n\nA 32 miles per gallon average is a good starting point, and it's even more impressive considering it's an improvement over your old car. You might want to consider tracking other metrics, such as:\n\n1. Fuel efficiency over time: See if your mileage improves or decreases over time, which can indicate potential issues with your car's engine, tires, or driving habits.\n2. Fuel efficiency in different conditions: Compare your mileage in city driving, highway driving, or in different weather conditions to see how they affect your car's performance.\n3. Maintenance and repair costs: Keep track of how much you're spending on maintenance, repairs, and replacement parts to get a better understanding of your car's overall cost of ownership.\n\nBy tracking these metrics, you'll be able to make data-driven decisions about your car's maintenance, optimize your driving habits, and potentially even save money on fuel costs.\n\nRemember, it's always a good idea to consult your car's owner's manual for recommended maintenance schedules, fuel efficiency tips, and other important information specific to your vehicle.\n\nKeep up the good work, and happy tracking!",
#         "has_answer": False
#     },
#     {
#         "role": "user",
#         "content": "I've also been using the rewards program at the Shell gas station near my office to earn points. Do you think it's worth it to focus on earning points at one gas station or to try to earn points across multiple stations?",
#         "has_answer": False
#     },
#     {
#         "role": "assistant",
#         "content": "That's a great question! Rewards programs can be a fantastic way to save money or earn perks on your gas purchases.\n\nThe strategy you choose depends on your personal preferences, driving habits, and the specific rewards programs offered by different gas stations. Here are some pros and cons of each approach to consider:\n\n**Focusing on one gas station:**\n\nPros:\n\n* You can accumulate points faster and reach rewards thresholds more quickly.\n* You might receive more targeted offers and promotions since you're a loyal customer.\n* It's easier to track your points and redeem rewards.\n\nCons:\n\n* You might be limited to a specific station or location, which could be inconvenient if you travel or have multiple cars.\n* You might miss out on better deals or rewards offered by other stations.\n\n**Earning points across multiple stations:**\n\nPros:\n\n* You can take advantage of better prices or promotions at different stations.\n* You're not tied to a single station, giving you more flexibility when traveling or filling up.\n* You can diversify your rewards earning potential across multiple programs.\n\nCons:\n\n* You might spread yourself too thin, making it harder to accumulate points quickly.\n* You'll need to keep track of multiple programs, which can be more complex.\n\nConsider the following questions to help you decide:\n\n1. Do you frequent a specific gas station regularly, making it convenient to focus on earning points there?\n2. Are the rewards offered by a particular station more valuable to you than others?\n3. Do you travel frequently or have multiple cars, making it beneficial to earn points across multiple stations?\n4. Are you willing to keep track of multiple rewards programs and adapt to changes in their terms and conditions?\n\nUltimately, you can experiment with both approaches and see what works best for your lifestyle and preferences. You might find that focusing on one station yields better rewards, or that earning points across multiple stations provides more flexibility and savings.\n\nRemember to always review the terms and conditions of each rewards program to ensure you understand how points are earned, redeemed, and expire. Happy rewards earning!",
#         "has_answer": False
#     },
#     {
#         "role": "user",
#         "content": "I've also been redeeming points from my credit card to get rewards. I recently redeemed 50,000 points to get a $500 gift card to a car accessories store. I used it to purchase a new car cover, floor mats, and a steering wheel cover, which I'm really happy with.",
#         "has_answer": False
#     },
#     {
#         "role": "assistant",
#         "content": "That's awesome! It sounds like you're making the most of your credit card rewards program!\n\nRedeeming 50,000 points for a $500 gift card is a great value, and it's even better that you were able to use it to get some practical and useful items for your car. A new car cover, floor mats, and steering wheel cover can really enhance the overall appearance and comfort of your vehicle.\n\nIt's great that you're thinking strategically about your rewards redemptions and using them to get something that brings you joy and value. Credit card rewards can be a fantastic way to offset the cost of everyday expenses or splurge on something special.\n\nDo you have a favorite way to earn or redeem credit card points, or any tips for maximizing your rewards earnings?\n\nAlso, how do you like your new car accessories? Are there any other car-related purchases or upgrades on your wishlist?",
#         "has_answer": False
#     },
#     {
#         "role": "user",
#         "content": "I'm really happy with my new car accessories, and they've added a nice touch to my car. I've been thinking about getting a car wax and detailing done soon, and I was wondering if you could help me find a good detailer in my area.",
#         "has_answer": False
#     },
#     {
#         "role": "assistant",
#         "content": "I'm glad to hear that you're happy with your new car accessories!\n\nAs for finding a good detailer in your area, I'd be happy to help. However, I'm a large language model, I don't have direct access to your location or personal information. But I can provide you with some general tips on how to find a reputable and reliable detailer in your area:\n\n1. Online Reviews: Check review websites such as Yelp, Google Reviews, or Facebook Reviews to see what other customers have to say about their experiences with local detailers.\n2. Ask for Referrals: Ask friends, family, or coworkers who have had their cars detailed recently for recommendations.\n3. Check Certifications: Look for detailers who are certified by organizations such as the International Detailing Association (IDA) or the National Institute of Automotive Service Excellence (ASE).\n4. Check Their Work: Ask to see before-and-after photos of their work or visit their shop to see the quality of their work firsthand.\n5. Get Quotes: Get quotes from several detailers and compare prices and services offered.\n\nBy following these steps, you should be able to find a reputable and reliable detailer in your area who can help you get your car looking its best.\n\nRemember to always do your research and choose a detailer who has a good reputation, provides quality work, and offers competitive pricing. Good luck in your search!",
#         "has_answer": False
#     }
# ]
# session = Session(session=sess, extract_keys=True, extract_facts=True, llm_extractor=test_llm_extractor)
# session.show_session()
# session.extract_session_keys(test_llm_extractor)
# session.extract_session_facts(test_llm_extractor)
# print(session)



conv=[
    [
        {
            "role": "user",
            "content": "I'm thinking of getting my car detailed soon. Do you know any good detailers in the area or have any recommendations? By the way, I just got my car serviced for the first time on March 15th, and it was a great experience.",
            "has_answer": True
        },
        {
            "role": "assistant",
            "content": "I'm happy to help you with your question! However, I'm a large language model, I don't have personal experiences or knowledge of specific locations or businesses. I can provide some general advice and suggestions on how to find a good detailer in your area.\n\nTo find a reputable and reliable detailer, you can try the following:\n\n1. Online Reviews: Check review websites such as Yelp, Google Reviews, or Facebook Reviews to see what other customers have to say about their experiences with local detailers.\n2. Ask for Referrals: Ask friends, family, or coworkers who have had their cars detailed recently for recommendations.\n3. Check Certifications: Look for detailers who are certified by organizations such as the International Detailing Association (IDA) or the National Institute of Automotive Service Excellence (ASE).\n4. Check Their Work: Ask to see before-and-after photos of their work or visit their shop to see the quality of their work firsthand.\n5. Get Quotes: Get quotes from several detailers and compare prices and services offered.\n\nAs for your car service experience on March 15th, I'm glad to hear it was a great experience! It's always important to keep your car well-maintained, and it sounds like you're taking good care of your vehicle.\n\nRemember to always do your research and choose a reputable detailer to ensure your car receives the best care possible. Good luck in your search!",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "I'll definitely check out those review websites and ask around for referrals. Do you think it's a good idea to get a wax and detailing done every 3-4 months to maintain the car's appearance and protect the paint?",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "Regular waxing and detailing can definitely help maintain your car's appearance and protect the paint. The frequency of waxing and detailing depends on several factors, including:\n\n1. Climate: If you live in an area with harsh weather conditions, such as heavy rain, snow, or intense sunlight, you may need to wax and detail more frequently (every 2-3 months).\n2. Driving habits: If you drive frequently, park outdoors, or drive on rough roads, you may need to wax and detail more often to remove dirt, grime, and contaminants that can damage the paint.\n3. Paint type: Some paint finishes, like clear coat or ceramic coat, may require more frequent waxing and detailing to maintain their appearance and protection.\n\nIn general, waxing every 3-4 months is a good rule of thumb for most cars. This frequency can help:\n\n1. Protect the paint from UV rays, oxidation, and fading.\n2. Remove dirt, grime, and contaminants that can scratch or damage the paint.\n3. Maintain a glossy, showroom-like appearance.\n4. Make future washing and maintenance easier.\n\nHowever, if you want to take your car's appearance to the next level, you may consider a more frequent detailing schedule, such as:\n\n* Washing every 1-2 weeks to remove dirt and grime.\n* Waxing every 2-3 months to provide an extra layer of protection.\n* Detailing every 3-4 months to deep clean the interior and exterior.\n\nRemember to also consider your car's specific needs, such as any special paint finishes or requirements.\n\nIt's great that you're thinking about maintaining your car's appearance and protection. Regular waxing and detailing can definitely help extend the life of your car's paint and keep it looking its best!",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "Yup, that makes sense. I've been keeping track of my gas mileage too, and I've been getting around 32 miles per gallon, which is better than my old car.",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "That's great to hear! Keeping track of your gas mileage is a great way to monitor your car's performance and make adjustments to your driving habits or maintenance routine as needed.\n\nA 32 miles per gallon average is a good starting point, and it's even more impressive considering it's an improvement over your old car. You might want to consider tracking other metrics, such as:\n\n1. Fuel efficiency over time: See if your mileage improves or decreases over time, which can indicate potential issues with your car's engine, tires, or driving habits.\n2. Fuel efficiency in different conditions: Compare your mileage in city driving, highway driving, or in different weather conditions to see how they affect your car's performance.\n3. Maintenance and repair costs: Keep track of how much you're spending on maintenance, repairs, and replacement parts to get a better understanding of your car's overall cost of ownership.\n\nBy tracking these metrics, you'll be able to make data-driven decisions about your car's maintenance, optimize your driving habits, and potentially even save money on fuel costs.\n\nRemember, it's always a good idea to consult your car's owner's manual for recommended maintenance schedules, fuel efficiency tips, and other important information specific to your vehicle.\n\nKeep up the good work, and happy tracking!",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "I've also been using the rewards program at the Shell gas station near my office to earn points. Do you think it's worth it to focus on earning points at one gas station or to try to earn points across multiple stations?",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "That's a great question! Rewards programs can be a fantastic way to save money or earn perks on your gas purchases.\n\nThe strategy you choose depends on your personal preferences, driving habits, and the specific rewards programs offered by different gas stations. Here are some pros and cons of each approach to consider:\n\n**Focusing on one gas station:**\n\nPros:\n\n* You can accumulate points faster and reach rewards thresholds more quickly.\n* You might receive more targeted offers and promotions since you're a loyal customer.\n* It's easier to track your points and redeem rewards.\n\nCons:\n\n* You might be limited to a specific station or location, which could be inconvenient if you travel or have multiple cars.\n* You might miss out on better deals or rewards offered by other stations.\n\n**Earning points across multiple stations:**\n\nPros:\n\n* You can take advantage of better prices or promotions at different stations.\n* You're not tied to a single station, giving you more flexibility when traveling or filling up.\n* You can diversify your rewards earning potential across multiple programs.\n\nCons:\n\n* You might spread yourself too thin, making it harder to accumulate points quickly.\n* You'll need to keep track of multiple programs, which can be more complex.\n\nConsider the following questions to help you decide:\n\n1. Do you frequent a specific gas station regularly, making it convenient to focus on earning points there?\n2. Are the rewards offered by a particular station more valuable to you than others?\n3. Do you travel frequently or have multiple cars, making it beneficial to earn points across multiple stations?\n4. Are you willing to keep track of multiple rewards programs and adapt to changes in their terms and conditions?\n\nUltimately, you can experiment with both approaches and see what works best for your lifestyle and preferences. You might find that focusing on one station yields better rewards, or that earning points across multiple stations provides more flexibility and savings.\n\nRemember to always review the terms and conditions of each rewards program to ensure you understand how points are earned, redeemed, and expire. Happy rewards earning!",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "I've also been redeeming points from my credit card to get rewards. I recently redeemed 50,000 points to get a $500 gift card to a car accessories store. I used it to purchase a new car cover, floor mats, and a steering wheel cover, which I'm really happy with.",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "That's awesome! It sounds like you're making the most of your credit card rewards program!\n\nRedeeming 50,000 points for a $500 gift card is a great value, and it's even better that you were able to use it to get some practical and useful items for your car. A new car cover, floor mats, and steering wheel cover can really enhance the overall appearance and comfort of your vehicle.\n\nIt's great that you're thinking strategically about your rewards redemptions and using them to get something that brings you joy and value. Credit card rewards can be a fantastic way to offset the cost of everyday expenses or splurge on something special.\n\nDo you have a favorite way to earn or redeem credit card points, or any tips for maximizing your rewards earnings?\n\nAlso, how do you like your new car accessories? Are there any other car-related purchases or upgrades on your wishlist?",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "I'm really happy with my new car accessories, and they've added a nice touch to my car. I've been thinking about getting a car wax and detailing done soon, and I was wondering if you could help me find a good detailer in my area.",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "I'm glad to hear that you're happy with your new car accessories!\n\nAs for finding a good detailer in your area, I'd be happy to help. However, I'm a large language model, I don't have direct access to your location or personal information. But I can provide you with some general tips on how to find a reputable and reliable detailer in your area:\n\n1. Online Reviews: Check review websites such as Yelp, Google Reviews, or Facebook Reviews to see what other customers have to say about their experiences with local detailers.\n2. Ask for Referrals: Ask friends, family, or coworkers who have had their cars detailed recently for recommendations.\n3. Check Certifications: Look for detailers who are certified by organizations such as the International Detailing Association (IDA) or the National Institute of Automotive Service Excellence (ASE).\n4. Check Their Work: Ask to see before-and-after photos of their work or visit their shop to see the quality of their work firsthand.\n5. Get Quotes: Get quotes from several detailers and compare prices and services offered.\n\nBy following these steps, you should be able to find a reputable and reliable detailer in your area who can help you get your car looking its best.\n\nRemember to always do your research and choose a detailer who has a good reputation, provides quality work, and offers competitive pricing. Good luck in your search!",
            "has_answer": False
        }
    ],
    [
        {
            "role": "user",
            "content": "I'm thinking of getting a car wax and detailing done soon. Can you give me some tips on what to look for when choosing a detailer?",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "Choosing the right detailer can make all the difference in getting your car looking its best. Here are some valuable tips to help you find a reliable and skilled detailer:\n\n1. **Check online reviews**: Look up reviews on platforms like Google, Yelp, Facebook, and other review sites to see what previous customers have to say about their experience with the detailer. Pay attention to the overall rating and read the comments to get an idea of their strengths and weaknesses.\n2. **Ask for referrals**: Ask friends, family, or coworkers who have had their cars detailed recently for recommendations. They can provide valuable insights and firsthand experience with the detailer.\n3. **Check their credentials**: Look for certifications from reputable organizations like the International Detailing Association (IDA) or the National Detailing Association (NDA). These certifications indicate that the detailer has undergone training and has a certain level of expertise.\n4. **Check their equipment and products**: A good detailer should have high-quality equipment and products to ensure a professional finish. Ask about the types of waxes, polishes, and cleaning solutions they use.\n5. **Inspect their work**: Ask to see before-and-after photos of their previous work or visit their shop to inspect the quality of their work firsthand. This will give you an idea of their attention to detail and level of craftsmanship.\n6. **Get a quote and ask about services**: Get a detailed quote that outlines the services they'll provide, including the type of wax, polish, or protective coating they'll use. Ask about any additional services they offer, such as interior detailing, headlight restoration, or paint correction.\n7. **Check their insurance and liability**: Make sure the detailer has liability insurance to protect your vehicle in case of damage or accidents.\n8. **Check their experience with your vehicle type**: If you have a specialty or high-performance vehicle, ensure the detailer has experience working with similar vehicles.\n9. **Check their turnaround time**: Find out how long the detailing process will take and whether they offer express or priority services if you're short on time.\n10. **Trust your instincts**: If something feels off or you don't click with the detailer, trust your instincts and look for someone else. You want to feel confident that your vehicle is in good hands.\n\nAdditionally, here are some red flags to watch out for:\n\n* Unrealistically low prices\n* Lack of transparency about their process or products\n* Poor communication or unresponsiveness\n* Unprofessional or dirty facilities\n* No clear warranty or guarantee\n\nBy following these tips, you'll be well on your way to finding a reliable and skilled detailer who can give your car the TLC it deserves.",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "I've been doing some research and found a local detailer with great reviews. I was thinking of taking my car there, but I'm also considering other options. By the way, I recently had an issue with my car's GPS system on 3/22, and I had to take it back to the dealership to get it fixed. They replaced the entire system, and now it's working flawlessly. Have you heard of any common issues with GPS systems in newer cars?",
            "has_answer": True
        },
        {
            "role": "assistant",
            "content": "That's great to hear that the dealership was able to fix your GPS issue! As for common issues with GPS systems in newer cars, yes, there are a few things to be aware of:\n\n1. **Software glitches**: Like any complex electronic system, GPS software can be prone to glitches, freezing, or crashing. These issues are often resolved with software updates, which can usually be done at the dealership or through the car's infotainment system.\n2. **Satellite signal interference**: GPS signals can be affected by tall buildings, tunnels, or heavy tree cover, leading to inaccurate or lost signals. This is usually a temporary issue, and the system will recover once the obstruction is cleared.\n3. **Map updates**: GPS systems rely on map data, which can become outdated if not updated regularly. This can lead to incorrect routing, missing roads, or inaccurate traffic information. Many manufacturers offer map updates, which can be done at the dealership or through the car's infotainment system.\n4. **Hardware failures**: In some cases, the GPS antenna or receiver can fail, causing the system to malfunction. This is less common but can require replacement of the affected component.\n5. **Integration issues**: Some GPS systems may not integrate seamlessly with other car systems, such as the infotainment system or smartphone connectivity. This can lead to issues like incorrect display information or failed voice commands.\n6. **Cybersecurity risks**: As with any connected system, GPS systems can be vulnerable to cyberattacks. Manufacturers are working to address these risks, but it's essential to keep your car's software up to date to minimize the risk of exploitation.\n\nIt's great that your dealership was able to resolve the issue with your GPS system. If you experience any further problems, don't hesitate to reach out to them or the manufacturer for assistance.\n\nRegarding the detailer you're considering, it's excellent that they have great reviews! Before making a final decision, you might want to ask them a few questions to ensure they're a good fit for your car's needs:\n\n1. What type of wax or protective coating do they use, and how long does it last?\n2. Do they use soft, microfiber cloths to avoid scratching the paint, and what kind of cleaning products do they use?\n3. Can they provide before-and-after photos of their work or show you a sample of their detailing process?\n4. Do they offer any guarantees or warranties on their work?\n5. How long will the detailing process take, and can they accommodate your schedule?\n\nBy asking these questions, you'll get a better sense of their expertise, attention to detail, and commitment to customer satisfaction.",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "I'll definitely ask those questions when I visit the detailer. And yeah, the GPS issue was a bit frustrating, but at least it was resolved quickly. By the way, do you know if there are any specific car wax products that are better suited for cars with advanced paint finishes, like my silver Honda Civic?",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "Advanced paint finishes, like those found on many modern vehicles, require special care to maintain their appearance and protect them from damage. When it comes to car wax, you'll want to look for products that are specifically designed for these types of finishes. Here are some tips to help you choose the right wax for your silver Honda Civic:\n\n1. **Look for waxes labeled as \"safe for advanced paint finishes\" or \"clear coat safe\"**: These waxes are formulated to work with the unique properties of advanced paint finishes, which can be more sensitive to certain chemicals.\n2. **Opt for a wax with a high Carnauba content**: Carnauba wax is a natural wax that provides an excellent barrier against the elements and is gentle on paint. Look for waxes with a high percentage of Carnauba (usually 20% or higher).\n3. **Consider a synthetic sealant or ceramic coating**: These products offer longer-lasting protection than traditional waxes and can provide an additional layer of defense against the elements. They're especially useful for cars with advanced paint finishes.\n4. **Avoid waxes with abrasive ingredients**: Some waxes contain abrasives that can scratch or mar advanced paint finishes. Look for waxes that are labeled as \"abrasive-free\" or \"non-abrasive.\"\n5. **Check the pH level**: A pH level close to neutral (around 7) is ideal for most paint finishes. Avoid waxes with high or low pH levels, as they can potentially damage the paint.\n\nSome popular car wax products that are suitable for advanced paint finishes include:\n\n* Meguiar's Ultimate Liquid Wax\n* Chemical Guys Car Wax\n* Autogeek Garage Line Car Wax\n* P21S Concours Carnauba Wax\n* Gtechniq Crystal Serum Ceramic Coating (for a longer-lasting, more protective option)\n\nRemember to always read the instructions and follow the manufacturer's recommendations for application and removal.\n\nWhen you visit the detailer, ask them about the type of wax they use and whether it's suitable for your car's advanced paint finish. A good detailer will be knowledgeable about the products they use and can recommend the best option for your vehicle.",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "I'll definitely ask about the wax they use and whether it's suitable for my car's paint finish. By the way, I've been keeping track of my gas mileage, and I've noticed that I've been getting an average of 32 miles per gallon, which is better than my old car. Do you have any tips on how to improve my gas mileage even further?",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "Congratulations on the improved gas mileage! 32 mpg is a great start, and with a few tweaks, you can optimize your fuel efficiency even further. Here are some tips to help you squeeze out a few more miles per gallon:\n\n1. **Regular maintenance**: Keep your car in top condition by regularly changing the oil, air filter, and spark plugs. A well-maintained engine will run more efficiently.\n2. **Tire pressure**: Properly inflated tires can improve gas mileage by up to 3%. Check your owner's manual for the recommended pressure and make sure to check your tires at least once a month.\n3. **Drive smoothly**: Aggressive driving, such as rapid acceleration and braking, can lower gas mileage. Drive smoothly, and use cruise control on the highway to maintain a constant speed.\n4. **Remove extra weight**: The more weight your car carries, the more energy it needs to move. Remove any unnecessary items from your trunk or back seat to improve gas mileage.\n5. **Use the correct oil**: Use the type of oil recommended by your car's manufacturer. Some cars are designed to run on synthetic oil, which can improve fuel efficiency.\n6. **Use gas-saving features**: Many modern cars come with features like automatic start/stop technology and eco-mode. Use these features to help improve gas mileage.\n7. **Plan your route**: Plan your route in advance to avoid traffic, construction, and other obstacles that can lower gas mileage.\n8. **Combine trips**: Try to combine errands and tasks to reduce the amount of time you spend driving.\n9. **Avoid idling**: If you're going to be stopped for more than 30 seconds, turn off your engine. Idling can waste a significant amount of fuel.\n10. **Monitor your gas cap**: A loose or damaged gas cap can cause gas to evaporate, reducing gas mileage. Make sure your gas cap is tight and in good condition.\n11. **Use fuel-efficient driving techniques**: Techniques like coasting to a stop, accelerating gradually, and using neutral while idling can all improve gas mileage.\n12. **Consider a fuel additive**: Certain fuel additives, like those that clean the engine or fuel system, can help improve gas mileage.\n\nRemember, even small changes to your driving habits and car maintenance can add up to make a big difference in your gas mileage. Keep track of your progress and see what works best for you!\n\nNow, back to your detailing appointment: don't forget to ask about any additional services they offer, such as interior cleaning or paint protection. A good detailer can help you maintain your car's appearance and resale value.",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "That's really helpful, thanks! I'll definitely keep those tips in mind to improve my gas mileage. And yeah, I'll ask about additional services like interior cleaning and paint protection when I visit the detailer. By the way, have you heard about any good car insurance companies that offer discounts for safe drivers? I just paid my premium online using my bank's mobile app, and I was happy to see that my premium had decreased by $20 per month due to my safe driving record.",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "Congratulations on the premium decrease! That's great to hear that your safe driving habits are being rewarded.\n\nThere are several car insurance companies that offer discounts for safe drivers. Here are a few popular ones:\n\n1. **State Farm**: State Farm offers a Drive Safe & Save program, which uses a mobile app to track your driving habits and rewards safe driving with discounts.\n2. **Progressive**: Progressive's Snapshot program uses a device that plugs into your car's OBD-II port to track your driving habits. Safe drivers can earn discounts of up to 30%.\n3. **Geico**: Geico offers a DriveEasy program, which uses a mobile app to track your driving habits and rewards safe driving with discounts.\n4. **Liberty Mutual**: Liberty Mutual's RightTrack program uses a mobile app to track your driving habits and rewards safe driving with discounts.\n5. **USAA**: USAA offers a SafePilot program, which uses a mobile app to track your driving habits and rewards safe driving with discounts.\n\nThese are just a few examples, and there are many other insurance companies that offer similar programs. It's always a good idea to shop around and compare rates and discounts from different insurers to find the best fit for your needs.\n\nIt's great that your bank's mobile app made it easy for you to pay your premium online. Many insurance companies and banks offer mobile apps that allow you to manage your policies, make payments, and even file claims on the go.\n\nRemember to always review your policy and coverage options regularly to ensure you're getting the best rates and coverage for your needs. And don't hesitate to ask your insurance agent or company about any discounts you may be eligible for, such as multi-car discounts, good student discounts, or professional association discounts.\n\nNow, back to your detailing appointment: don't forget to ask about any special promotions or discounts they may be offering. Some detailers may offer package deals or loyalty programs that can help you save money in the long run.",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "That's really helpful, thanks! I'll definitely look into those insurance companies and programs. And yeah, I'll ask about promotions or discounts when I visit the detailer. By the way, I recently helped my friend Emily move to her new apartment, and I used my car to transport some of her furniture. She was very appreciative, and we had a great time catching up during the move. Do you have any tips on how to protect my car's interior from wear and tear, especially when transporting large or heavy items?",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "That's great that you were able to help your friend Emily with her move! It's always nice to lend a helping hand, and it's even better when you can catch up with friends in the process.\n\nProtecting your car's interior from wear and tear is essential, especially when transporting large or heavy items. Here are some tips to help you keep your car's interior looking its best:\n\n1. **Use floor mats**: Invest in a good set of floor mats that can withstand heavy foot traffic and protect your car's carpets from dirt, mud, and stains.\n2. **Protect your seats**: Use seat covers or throw blankets to protect your car's seats from dirt, stains, and tears. You can also use seat shields or seat protectors for added protection.\n3. **Cover your cargo area**: Use a cargo mat or cargo liner to protect your car's trunk or cargo area from dirt, dust, and scratches. This is especially important when transporting large or heavy items.\n4. **Secure your cargo**: Make sure to secure your cargo properly using ropes, straps, or nets to prevent items from shifting around and causing damage to your car's interior.\n5. **Clean up spills immediately**: If you spill something in your car, clean it up immediately to prevent stains from setting in. Use a cleaning solution specifically designed for your car's interior surfaces.\n6. **Vacuum regularly**: Vacuum your car's interior regularly to remove dirt, dust, and debris that can cause wear and tear on your car's surfaces.\n7. **Use a dashboard protector**: Use a dashboard protector to prevent your car's dashboard from cracking or fading due to exposure to sunlight.\n8. **Protect your doors**: Use door sill protectors to prevent scratches and scuffs on your car's doors.\n9. **Use a trunk organizer**: Use a trunk organizer to keep your car's trunk or cargo area organized and prevent items from shifting around.\n10. **Regularly condition your leather**: If your car has leather seats, regularly condition them to keep them soft and supple.\n\nBy following these tips, you can help protect your car's interior from wear and tear, even when transporting large or heavy items.\n\nNow, back to your detailing appointment: don't forget to ask about any interior detailing services they offer, such as upholstery cleaning, carpet shampooing, or glass treatment. A good detailer can help you keep your car's interior looking its best.",
            "has_answer": False
        }
    ],
    [
        {
            "role": "user",
            "content": "I'm thinking of planning a road trip soon and I'm trying to figure out the best route to take. I've got a new car, a silver Honda Civic that I bought on February 10th, and I want to make sure I'm getting the most out of its features, like the GPS system. Can you help me with that?",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "Congratulations on your new car! I'd be happy to help you plan a road trip and make the most of your Honda Civic's GPS system.\n\nTo get started, can you please provide me with some more details about your trip? Here are a few questions to consider:\n\n1. Where are you starting and ending your trip?\n2. How many days do you have for the trip?\n3. What type of scenery or attractions are you interested in seeing along the way (e.g. national parks, beaches, cities, mountains)?\n4. Are there any specific routes or highways you're interested in taking or avoiding?\n5. Do you have any preferences for accommodations or dining options along the way?\n6. Are there any particular features of your GPS system you're looking to utilize (e.g. traffic updates, lane guidance, voice commands)?\n\nOnce I have a better understanding of your trip plans, I can provide you with some personalized route suggestions and tips for getting the most out of your GPS system.",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "I'm planning a 5-day trip from my hometown to Yellowstone National Park. I've always wanted to visit and I think the scenic route would be amazing. I'm open to any suggestions for accommodations and dining options, but I'd prefer budget-friendly choices. I'm also interested in using the lane departure warning and blind spot detection features of my GPS system to ensure a safe drive.",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "What a fantastic destination! Yellowstone is a breathtaking national park, and I'm excited to help you plan your trip.\n\nFor a scenic route to Yellowstone, I recommend taking the \"Beartooth Highway\" route, which is considered one of the most beautiful drives in the country. Here's a suggested 5-day itinerary for you:\n\nDay 1: Hometown to Mount Rushmore, SD (approx. 560 miles, 8 hours)\n\n* Take I-90 W from your hometown to Mount Rushmore National Memorial in South Dakota.\n* Visit the iconic presidential monument and explore the surrounding Black Hills.\n* Overnight in nearby Rapid City, SD. Budget-friendly options include the Rushmore Hotel & Suites or the Microtel Inn & Suites.\n\nDay 2: Mount Rushmore to Cody, WY (approx. 340 miles, 5 hours)\n\n* Continue west on I-90 W to Sheridan, WY, then take US-16 W to Cody, WY.\n* Visit the Buffalo Bill Center of the West, a museum showcasing Western art and history.\n* Overnight in Cody. Budget-friendly options include the Cody Hotel or the Comfort Inn Cody.\n\nDay 3: Cody to Yellowstone National Park, WY (approx. 52 miles, 1 hour)\n\n* Take US-16 W to the east entrance of Yellowstone National Park.\n* Drive through the park's scenic routes, such as the Beartooth Highway (US-212), and explore the park's geysers, hot springs, and wildlife.\n* Overnight in the park at one of the many campgrounds or lodges, such as the Grant Village Campground or the Lake Yellowstone Hotel.\n\nDay 4: Yellowstone National Park\n\n* Spend the day exploring Yellowstone's attractions, such as Old Faithful, the Grand Prismatic Spring, and the Upper and Lower Yellowstone Falls.\n* Consider taking a guided tour or hike to get the most out of your park experience.\n\nDay 5: Yellowstone National Park to Hometown (approx. 560 miles, 8 hours)\n\n* Depart Yellowstone through the east entrance and retrace your route back to your hometown.\n\nTo make the most of your GPS system's features, be sure to:\n\n* Update your GPS system's maps to ensure you have the latest road data.\n* Enable lane departure warning and blind spot detection to help you stay safe on the road.\n* Use voice commands to navigate and set destinations, allowing you to keep your eyes on the road.\n* Take regular breaks to rest and stretch your legs, especially on long driving days.\n\nBudget-friendly dining options along the way include:\n\n* Fast food chains like Subway or McDonald's\n* Roadside diners like the Cowboy Caf\u00e9 in Cody or the Irma Hotel's restaurant in Cody\n* Grocery stores like Albertsons or Safeway for picnic supplies\n\nRemember to check the road conditions and weather forecast before you leave, and consider downloading a traffic and weather app to help you navigate any potential road closures or construction.\n\nHave a safe and enjoyable trip to Yellowstone!",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "I'm excited about this itinerary, but I'm a bit concerned about the fuel efficiency of my new car on these long drives. Can you tell me how to optimize my Honda Civic's fuel efficiency, especially on the highway? I've been getting around 32 miles per gallon in the city, but I'm not sure what to expect on these long road trips.",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "Fuel efficiency is a great concern, especially on long road trips. I'm happy to help you optimize your Honda Civic's fuel efficiency on the highway.\n\nThe Honda Civic is known for its excellent fuel economy, and with some simple tips, you can maximize its fuel efficiency on the highway. Here are some tips to help you get the best mileage:\n\n1. **Maintain a consistent speed**: Try to maintain a steady speed, as frequent acceleration and braking can lower your fuel efficiency. Use cruise control on the highway to help you maintain a constant speed.\n2. **Drive at moderate speeds**: The ideal speed for fuel efficiency varies, but generally, it's between 60-70 mph. Avoid speeding, as it can decrease your fuel efficiency significantly.\n3. **Use the correct gear**: Make sure you're in the correct gear for the speed you're driving. Shifting into higher gears at lower speeds can help improve fuel efficiency.\n4. **Avoid idling**: If you're going to be stopped for more than 30 seconds, turn off your engine. Idling can waste a significant amount of fuel.\n5. **Use air conditioning wisely**: While using the air conditioning can be comfortable on hot days, it can decrease your fuel efficiency. Try rolling down your windows at lower speeds to get some fresh air and reduce AC usage.\n6. **Check your tire pressure**: Properly inflated tires can improve your fuel efficiency. Underinflated tires can increase rolling resistance, which can decrease fuel efficiency.\n7. **Remove extra weight**: The more weight your car is carrying, the more fuel it'll consume. Remove any unnecessary items from your trunk or back seat to improve fuel efficiency.\n8. **Use fuel-efficient driving modes**: Many modern cars, including the Honda Civic, come with fuel-efficient driving modes. Enable this mode to optimize your car's engine performance, transmission, and other systems for better fuel efficiency.\n9. **Monitor your fuel efficiency**: Keep an eye on your fuel efficiency using your car's trip computer or a fuel log app. This will help you identify areas for improvement and develop good driving habits.\n10. **Regular maintenance**: Regular tune-ups, oil changes, and filter replacements can help maintain your car's fuel efficiency.\n\nAs for your specific question, the Honda Civic's fuel efficiency on the highway varies depending on the trim level and transmission type. However, according to the EPA, the 2022 Honda Civic sedan with a CVT transmission gets an estimated:\n\n* 32 mpg city\n* 42 mpg highway\n* 36 mpg combined\n\nBased on these estimates, you can expect to get around 38-40 mpg on the highway, assuming you're driving moderately and following the tips above. However, this can vary depending on your specific driving habits, road conditions, and weather.\n\nRemember, every little bit helps, and by following these tips, you can optimize your Honda Civic's fuel efficiency and enjoy a more cost-effective road trip to Yellowstone!",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "I've been using the Shell gas station near my office to fill up, and I've been earning points through their rewards program. Do you think I should fill up at Shell stations along the way to maximize my rewards or should I look for cheaper gas prices at other stations?",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "Loyalty programs can be a great way to save money and earn rewards. In your case, since you've already been earning points through Shell's rewards program, it's definitely worth considering filling up at Shell stations along the way to maximize your rewards.\n\nHere are some pros and cons to help you decide:\n\n**Pros of filling up at Shell stations:**\n\n1. **Maximize rewards**: By continuing to fill up at Shell stations, you'll earn more points and get closer to redeeming rewards, such as discounts on fuel, merchandise, or even cashback.\n2. **Consistency**: You're already familiar with Shell's rewards program, so it's easy to stick with what you know and continue earning points.\n3. **Convenience**: Shell has a large network of stations, so you're likely to find one near your route.\n\n**Cons of filling up at Shell stations:**\n\n1. **Potential higher prices**: Shell prices might be higher than those at other stations, which could offset the benefits of earning rewards.\n2. **Limited flexibility**: If you're only looking for Shell stations, you might miss out on cheaper options or more convenient locations.\n\n**Alternatives to consider:**\n\n1. **Gas price comparison apps**: Use apps like GasBuddy, Waze, or Google Maps to find the cheapest gas prices along your route. This way, you can compare prices and make informed decisions.\n2. **Other loyalty programs**: If you find a station with a better price, consider joining their loyalty program. You might earn rewards or discounts that outweigh the benefits of sticking with Shell.\n3. **Discount stations**: Look for stations that offer discounts for cash payments, military personnel, or other groups. These can be a great way to save money, even if you're not earning rewards.\n\nUltimately, the decision depends on your priorities. If you value the rewards and convenience of Shell's program, it might be worth sticking with them. However, if you're looking to save money on gas prices, exploring other options could be a better choice.\n\nYou could also consider a hybrid approach:\n\n* Fill up at Shell stations when they're convenient and the prices are competitive.\n* Use gas price comparison apps to find cheaper options when Shell prices are high.\n* Take advantage of other loyalty programs or discounts when available.\n\nBy being flexible and aware of your options, you can make the most of your road trip and your rewards program.",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "I think I'll try to balance my loyalty to Shell with finding cheaper options along the way. I'll use gas price comparison apps to find the best deals and fill up at Shell stations when they're competitive. That way, I can earn rewards while also saving money on gas.",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "That's a great approach! By balancing your loyalty to Shell with finding cheaper options, you can maximize your rewards while also saving money on gas. Using gas price comparison apps will help you make informed decisions and find the best deals along your route.\n\nRemember, every little bit counts, and saving a few dollars on gas here and there can add up over the course of your road trip. By being flexible and adaptable, you can enjoy the benefits of Shell's rewards program while also being mindful of your budget.\n\nHere are a few more tips to help you make the most of your approach:\n\n1. **Set a price threshold**: Decide on a maximum price you're willing to pay for gas and stick to it. If you find a Shell station above that threshold, look for a cheaper option.\n2. **Use cashback apps**: In addition to gas price comparison apps, consider using cashback apps like Ibotta, Fetch Rewards, or Checkout 51 that offer rebates on gas purchases.\n3. **Take advantage of discounts**: If you have a credit card or membership that offers discounts on gas, be sure to use them.\n4. **Fuel up during off-peak hours**: Avoid filling up during peak hours (usually morning and evening rush hour) when prices might be higher.\n5. **Don't sacrifice convenience**: While saving money is important, don't sacrifice convenience for the sake of a few cents. If you're tired or need a break, stop at a convenient station, even if it's not the cheapest option.\n\nBy following these tips and balancing your loyalty to Shell with finding cheaper options, you'll be well on your way to a successful and cost-effective road trip to Yellowstone!",
            "has_answer": False
        },
        {
            "role": "user",
            "content": "I'm thinking of getting a car wax and detailing done before my trip to protect my car's paint job and interior. Do you have any recommendations for car wax and detailing products or services?",
            "has_answer": False
        },
        {
            "role": "assistant",
            "content": "Getting a car wax and detailing done before your trip is an excellent idea. It'll not only protect your car's paint job and interior but also give you a sense of pride and confidence on the road.\n\nFor car wax and detailing products, I recommend considering the following:\n\n1. **Meguiar's**: A popular and well-respected brand in the car care industry, Meguiar's offers a wide range of products, including waxes, polishes, and cleaners.\n2. **Chemical Guys**: Known for their high-quality, detailing-focused products, Chemical Guys offers a variety of waxes, sealants, and cleaners that are popular among car enthusiasts.\n3. **Autoglym**: A premium brand that offers high-end car care products, Autoglym is a great choice for those who want the best protection and shine for their vehicle.\n\nAs for detailing services, you can consider the following options:\n\n1. **Local car wash and detailing shops**: Look for reputable, well-reviewed shops in your area that offer detailing services. They'll have the necessary equipment and expertise to give your car a thorough cleaning and protection.\n2. **Mobile detailing services**: If you're short on time or prefer the convenience, consider mobile detailing services that come to your location. They'll bring their equipment and expertise to you.\n3. **DIY**: If you're comfortable with detailing your car yourself, you can purchase the necessary products and follow online tutorials or guides to achieve a professional-looking finish.\n\nSome popular detailing services to consider:\n\n1. **Ceramic coating**: A premium protection option that provides an invisible barrier against the elements, ceramic coating can last for months or even years.\n2. **Paint correction**: A service that removes imperfections and scratches, paint correction can leave your car's paint job looking like new.\n3. **Interior detailing**: A thorough cleaning and protection of your car's interior, including upholstery, carpets, and surfaces.\n\nBefore choosing a detailing service or product, be sure to:\n\n1. **Read reviews**: Check online reviews from multiple sources to ensure the product or service is reputable and effective.\n2. **Ask questions**: If you're unsure about a product or service, ask questions to ensure you understand what you're getting.\n3. **Check prices**: Compare prices and services to find the best value for your money.\n\nBy investing in a good car wax and detailing, you'll be protecting your car's paint job and interior, ensuring it looks great and stays protected on your road trip to Yellowstone!",
            "has_answer": False
        }
    ]
]

id_list=[
    "answer_4be1b6b4_2",
    "answer_4be1b6b4_3",
    "answer_4be1b6b4_1"
]
date_list=[
    "2023/04/10 (Mon) 17:50",
    "2023/04/10 (Mon) 14:47",
    "2023/04/10 (Mon) 17:15"
]

# conversation = Conversation(conv, id_list, date_list, date="现在", id='0597')
# print(conversation)

# conversation.extract_conversation_facts(test_llm_extractor)
# conversation.extract_conversation_keys(test_llm_extractor)

# print(conversation)
# conversation.show_conversation()