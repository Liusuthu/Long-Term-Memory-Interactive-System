import json

longmemeval_m_path = "./data/longmemeval_m"
longmemeval_s_path = "./data/longmemeval_s"
longmemeval_oracle_path = "./data/longmemeval_oracle"

def get_turns_sum(session):
    turns_sum = 0
    for chat in session:
        turns_sum += len(chat)/2
    return turns_sum


# Basic Information About longmemeval_s
print("-"*20+" longmemeval_s "+"-"*20)
with open(longmemeval_s_path, "r") as f:
    longmemeval_s = json.load(f)
    print("Number of questions: ",len(longmemeval_s))
    single_session_user_cnt = 0
    single_session_assistant_cnt = 0
    single_session_preference_cnt = 0
    multi_session_cnt = 0
    temporal_reasoning_cnt = 0
    knowledge_update_cnt = 0
    abstention_cnt = 0

    single_session_user_session_sum = 0
    single_session_assistant_session_sum = 0
    single_session_preference_session_sum = 0
    multi_session_session_sum = 0
    temporal_reasoning_session_sum = 0
    knowledge_update_session_sum = 0
    abstention_session_sum = 0

    single_session_user_evidence_session_sum = 0
    single_session_assistant_evidence_session_sum = 0
    single_session_preference_evidence_session_sum = 0
    multi_session_evidence_session_sum = 0
    temporal_reasoning_evidence_session_sum = 0
    knowledge_update_evidence_session_sum = 0
    abstention_evidence_session_sum = 0
    
    single_session_user_turns_sum = 0
    single_session_assistant_turns_sum = 0
    single_session_preference_turns_sum = 0
    multi_session_turns_sum = 0
    temporal_reasoning_turns_sum = 0
    knowledge_update_turns_sum = 0
    abstention_turns_sum = 0

    for item in longmemeval_s:
        if item["question_type"] == "single-session-user":
            single_session_user_cnt += 1
            single_session_user_session_sum += len(item["haystack_session_ids"])
            single_session_user_evidence_session_sum += len(item["answer_session_ids"])
            single_session_user_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "single-session-assistant":
            single_session_assistant_cnt += 1
            single_session_assistant_session_sum += len(item["haystack_session_ids"])
            single_session_assistant_evidence_session_sum += len(item["answer_session_ids"])
            single_session_assistant_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "single-session-preference":  
            single_session_preference_cnt += 1
            single_session_preference_session_sum += len(item["haystack_session_ids"])
            single_session_preference_evidence_session_sum += len(item["answer_session_ids"])
            single_session_preference_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "multi-session":
            multi_session_cnt += 1
            multi_session_session_sum += len(item["haystack_session_ids"])
            multi_session_evidence_session_sum += len(item["answer_session_ids"])
            multi_session_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "temporal-reasoning":
            temporal_reasoning_cnt += 1
            temporal_reasoning_session_sum += len(item["haystack_session_ids"])
            temporal_reasoning_evidence_session_sum += len(item["answer_session_ids"])
            temporal_reasoning_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "knowledge-update":
            knowledge_update_cnt += 1
            knowledge_update_session_sum += len(item["haystack_session_ids"])
            knowledge_update_evidence_session_sum += len(item["answer_session_ids"])
            knowledge_update_turns_sum += get_turns_sum(item["haystack_sessions"])
        if "_abs" in item["question_id"]:
            abstention_cnt += 1
            abstention_session_sum += len(item["haystack_session_ids"])
            abstention_evidence_session_sum += len(item["answer_session_ids"])
            abstention_turns_sum += get_turns_sum(item["haystack_sessions"])
    print("\tDistrubution of question types")
    print("\t\tsingle_session_user: ", single_session_user_cnt)
    print("\t\tsingle_session_assistant: ", single_session_assistant_cnt)
    print("\t\tsingle_session_preference: ", single_session_preference_cnt)
    print("\t\tmulti_session: ", multi_session_cnt)
    print("\t\ttemporal_reasoning: ", temporal_reasoning_cnt)
    print("\t\tknowledge_update: ", knowledge_update_cnt)
    print("\t\tabstention: ", abstention_cnt)
    print("\tAverage number of sessions in each question type")
    print("\t\tsingle_session_user: ", single_session_user_session_sum/single_session_user_cnt)
    print("\t\tsingle_session_assistant: ", single_session_assistant_session_sum/single_session_assistant_cnt)
    print("\t\tsingle_session_preference: ", single_session_preference_session_sum/single_session_preference_cnt)
    print("\t\tmulti_session: ", multi_session_session_sum/multi_session_cnt)
    print("\t\ttemporal_reasoning: ", temporal_reasoning_session_sum/temporal_reasoning_cnt)
    print("\t\tknowledge_update: ", knowledge_update_session_sum/knowledge_update_cnt)
    print("\t\tabstention: ", abstention_session_sum/abstention_cnt)
    print("\tAverage number of evidence sessions in each question type")
    print("\t\tsingle_session_user: ", single_session_user_evidence_session_sum/single_session_user_cnt)
    print("\t\tsingle_session_assistant: ", single_session_assistant_evidence_session_sum/single_session_assistant_cnt)
    print("\t\tsingle_session_preference: ", single_session_preference_evidence_session_sum/single_session_preference_cnt)
    print("\t\tmulti_session: ", multi_session_evidence_session_sum/multi_session_cnt)
    print("\t\ttemporal_reasoning: ", temporal_reasoning_evidence_session_sum/temporal_reasoning_cnt)
    print("\t\tknowledge_update: ", knowledge_update_evidence_session_sum/knowledge_update_cnt)
    print("\t\tabstention: ", abstention_evidence_session_sum/abstention_cnt)
    print("\tAverage number of turns in each question type")
    print("\t\tsingle_session_user: ", single_session_user_turns_sum/single_session_user_cnt)
    print("\t\tsingle_session_assistant: ", single_session_assistant_turns_sum/single_session_assistant_cnt)
    print("\t\tsingle_session_preference: ", single_session_preference_turns_sum/single_session_preference_cnt)
    print("\t\tmulti_session: ", multi_session_turns_sum/multi_session_cnt)
    print("\t\ttemporal_reasoning: ", temporal_reasoning_turns_sum/temporal_reasoning_cnt)
    print("\t\tknowledge_update: ", knowledge_update_turns_sum/knowledge_update_cnt)
    print("\t\tabstention: ", abstention_turns_sum/abstention_cnt)
    print("\tAverage number of turns per session in each question type")
    print("\t\tsingle_session_user: ", single_session_user_turns_sum/single_session_user_session_sum)
    print("\t\tsingle_session_assistant : ", single_session_assistant_turns_sum/single_session_assistant_session_sum)
    print("\t\tsingle_session_preference: ", single_session_preference_turns_sum/single_session_preference_session_sum)
    print("\t\tmulti_session: ", multi_session_turns_sum/multi_session_session_sum)
    print("\t\ttemporal_reasoning: ", temporal_reasoning_turns_sum/temporal_reasoning_session_sum)
    print("\t\tknowledge_update: ", knowledge_update_turns_sum/knowledge_update_session_sum)
    print("\t\tabstention: ", abstention_turns_sum/abstention_session_sum)


# Basic Information About longmemeval_oracle
print("-"*20+" longmemeval_oracle "+"-"*20)
with open(longmemeval_oracle_path, "r") as f:
    longmemeval_oracle = json.load(f)
    print("Number of questions: ",len(longmemeval_oracle))
    single_session_user_cnt = 0
    single_session_assistant_cnt = 0
    single_session_preference_cnt = 0
    multi_session_cnt = 0
    temporal_reasoning_cnt = 0
    knowledge_update_cnt = 0
    abstention_cnt = 0

    single_session_user_evidence_session_sum = 0
    single_session_assistant_evidence_session_sum = 0
    single_session_preference_evidence_session_sum = 0
    multi_session_evidence_session_sum = 0
    temporal_reasoning_evidence_session_sum = 0
    knowledge_update_evidence_session_sum = 0
    abstention_evidence_session_sum = 0

    single_session_user_turns_sum = 0
    single_session_assistant_turns_sum = 0
    single_session_preference_turns_sum = 0
    multi_session_turns_sum = 0
    temporal_reasoning_turns_sum = 0
    knowledge_update_turns_sum = 0
    abstention_turns_sum = 0
    for item in longmemeval_oracle:
        if item["question_type"] == "single-session-user":
            single_session_user_cnt += 1
            single_session_user_evidence_session_sum += len(item["answer_session_ids"])
            single_session_user_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "single-session-assistant":
            single_session_assistant_cnt += 1
            single_session_assistant_evidence_session_sum += len(item["answer_session_ids"])
            single_session_assistant_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "single-session-preference":  
            single_session_preference_cnt += 1
            single_session_preference_evidence_session_sum += len(item["answer_session_ids"])
            single_session_preference_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "multi-session":
            multi_session_cnt += 1
            multi_session_evidence_session_sum += len(item["answer_session_ids"])
            multi_session_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "temporal-reasoning":
            temporal_reasoning_cnt += 1
            temporal_reasoning_evidence_session_sum += len(item["answer_session_ids"])
            temporal_reasoning_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "knowledge-update":
            knowledge_update_cnt += 1
            knowledge_update_evidence_session_sum += len(item["answer_session_ids"])
            knowledge_update_turns_sum += get_turns_sum(item["haystack_sessions"])
        if "_abs" in item["question_id"]:
            abstention_cnt += 1
            abstention_evidence_session_sum += len(item["answer_session_ids"])
            abstention_turns_sum += get_turns_sum(item["haystack_sessions"])
    print("\tsingle_session_user: ", single_session_user_cnt)
    print("\t\tsingle_session_assistant: ", single_session_assistant_cnt)
    print("\t\tsingle_session_preference: ", single_session_preference_cnt)
    print("\t\tmulti_session: ", multi_session_cnt)
    print("\t\ttemporal_reasoning: ", temporal_reasoning_cnt)
    print("\t\tknowledge_update: ", knowledge_update_cnt)
    print("\t\tabstention: ", abstention_cnt)
    print("\tAverage number of turns per evidence session in each question type")
    print("\t\tsingle_session_user: ", single_session_user_turns_sum/single_session_user_evidence_session_sum)
    print("\t\tsingle_session_assistant : ", single_session_assistant_turns_sum/single_session_assistant_evidence_session_sum)
    print("\t\tsingle_session_preference: ", single_session_preference_turns_sum/single_session_preference_evidence_session_sum)
    print("\t\tmulti_session: ", multi_session_turns_sum/multi_session_evidence_session_sum)
    print("\t\ttemporal_reasoning: ", temporal_reasoning_turns_sum/temporal_reasoning_evidence_session_sum)
    print("\t\tknowledge_update: ", knowledge_update_turns_sum/knowledge_update_evidence_session_sum)
    print("\t\tabstention: ", abstention_turns_sum/abstention_evidence_session_sum)


# Basic Information About longmemeval_m
print("-"*20+" longmemeval_m "+"-"*20)
with open(longmemeval_m_path, "r") as f:
    longmemeval_m = json.load(f)
    print("Number of questions: ", len(longmemeval_m))
    single_session_user_cnt = 0
    single_session_assistant_cnt = 0
    single_session_preference_cnt = 0
    multi_session_cnt = 0
    temporal_reasoning_cnt = 0
    knowledge_update_cnt = 0
    abstention_cnt = 0

    single_session_user_session_sum = 0
    single_session_assistant_session_sum = 0
    single_session_preference_session_sum = 0
    multi_session_session_sum = 0
    temporal_reasoning_session_sum = 0
    knowledge_update_session_sum = 0
    abstention_session_sum = 0

    single_session_user_evidence_session_sum = 0
    single_session_assistant_evidence_session_sum = 0
    single_session_preference_evidence_session_sum = 0
    multi_session_evidence_session_sum = 0
    temporal_reasoning_evidence_session_sum = 0
    knowledge_update_evidence_session_sum = 0
    abstention_evidence_session_sum = 0
    
    single_session_user_turns_sum = 0
    single_session_assistant_turns_sum = 0
    single_session_preference_turns_sum = 0
    multi_session_turns_sum = 0
    temporal_reasoning_turns_sum = 0
    knowledge_update_turns_sum = 0
    abstention_turns_sum = 0

    for item in longmemeval_m:
        if item["question_type"] == "single-session-user":
            single_session_user_cnt += 1
            single_session_user_session_sum += len(item["haystack_session_ids"])
            single_session_user_evidence_session_sum += len(item["answer_session_ids"])
            single_session_user_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "single-session-assistant":
            single_session_assistant_cnt += 1
            single_session_assistant_session_sum += len(item["haystack_session_ids"])
            single_session_assistant_evidence_session_sum += len(item["answer_session_ids"])
            single_session_assistant_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "single-session-preference":  
            single_session_preference_cnt += 1
            single_session_preference_session_sum += len(item["haystack_session_ids"])
            single_session_preference_evidence_session_sum += len(item["answer_session_ids"])
            single_session_preference_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "multi-session":
            multi_session_cnt += 1
            multi_session_session_sum += len(item["haystack_session_ids"])
            multi_session_evidence_session_sum += len(item["answer_session_ids"])
            multi_session_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "temporal-reasoning":
            temporal_reasoning_cnt += 1
            temporal_reasoning_session_sum += len(item["haystack_session_ids"])
            temporal_reasoning_evidence_session_sum += len(item["answer_session_ids"])
            temporal_reasoning_turns_sum += get_turns_sum(item["haystack_sessions"])
        elif item["question_type"] == "knowledge-update":
            knowledge_update_cnt += 1
            knowledge_update_session_sum += len(item["haystack_session_ids"])
            knowledge_update_evidence_session_sum += len(item["answer_session_ids"])
            knowledge_update_turns_sum += get_turns_sum(item["haystack_sessions"])
        if "_abs" in item["question_id"]:
            abstention_cnt += 1
            abstention_session_sum += len(item["haystack_session_ids"])
            abstention_evidence_session_sum += len(item["answer_session_ids"])
            abstention_turns_sum += get_turns_sum(item["haystack_sessions"])
    print("\tDistrubution of question types")
    print("\t\tsingle_session_user: ", single_session_user_cnt)
    print("\t\tsingle_session_assistant: ", single_session_assistant_cnt)
    print("\t\tsingle_session_preference: ", single_session_preference_cnt)
    print("\t\tmulti_session: ", multi_session_cnt)
    print("\t\ttemporal_reasoning: ", temporal_reasoning_cnt)
    print("\t\tknowledge_update: ", knowledge_update_cnt)
    print("\t\tabstention: ", abstention_cnt)
    print("\tAverage number of sessions in each question type")
    print("\t\tsingle_session_user: ", single_session_user_session_sum/single_session_user_cnt)
    print("\t\tsingle_session_assistant: ", single_session_assistant_session_sum/single_session_assistant_cnt)
    print("\t\tsingle_session_preference: ", single_session_preference_session_sum/single_session_preference_cnt)
    print("\t\tmulti_session: ", multi_session_session_sum/multi_session_cnt)
    print("\t\ttemporal_reasoning: ", temporal_reasoning_session_sum/temporal_reasoning_cnt)
    print("\t\tknowledge_update: ", knowledge_update_session_sum/knowledge_update_cnt)
    print("\t\tabstention: ", abstention_session_sum/abstention_cnt)
    print("\tAverage number of evidence sessions in each question type")
    print("\t\tsingle_session_user: ", single_session_user_evidence_session_sum/single_session_user_cnt)
    print("\t\tsingle_session_assistant: ", single_session_assistant_evidence_session_sum/single_session_assistant_cnt)
    print("\t\tsingle_session_preference: ", single_session_preference_evidence_session_sum/single_session_preference_cnt)
    print("\t\tmulti_session: ", multi_session_evidence_session_sum/multi_session_cnt)
    print("\t\ttemporal_reasoning: ", temporal_reasoning_evidence_session_sum/temporal_reasoning_cnt)
    print("\t\tknowledge_update: ", knowledge_update_evidence_session_sum/knowledge_update_cnt)
    print("\t\tabstention: ", abstention_evidence_session_sum/abstention_cnt)
    print("\tAverage number of turns in each question type")
    print("\t\tsingle_session_user: ", single_session_user_turns_sum/single_session_user_cnt)
    print("\t\tsingle_session_assistant: ", single_session_assistant_turns_sum/single_session_assistant_cnt)
    print("\t\tsingle_session_preference: ", single_session_preference_turns_sum/single_session_preference_cnt)
    print("\t\tmulti_session: ", multi_session_turns_sum/multi_session_cnt)
    print("\t\ttemporal_reasoning: ", temporal_reasoning_turns_sum/temporal_reasoning_cnt)
    print("\t\tknowledge_update: ", knowledge_update_turns_sum/knowledge_update_cnt)
    print("\t\tabstention: ", abstention_turns_sum/abstention_cnt)
    print("\tAverage number of turns per session in each question type")
    print("\t\tsingle_session_user: ", single_session_user_turns_sum/single_session_user_session_sum)
    print("\t\tsingle_session_assistant : ", single_session_assistant_turns_sum/single_session_assistant_session_sum)
    print("\t\tsingle_session_preference: ", single_session_preference_turns_sum/single_session_preference_session_sum)
    print("\t\tmulti_session: ", multi_session_turns_sum/multi_session_session_sum)
    print("\t\ttemporal_reasoning: ", temporal_reasoning_turns_sum/temporal_reasoning_session_sum)
    print("\t\tknowledge_update: ", knowledge_update_turns_sum/knowledge_update_session_sum)
    print("\t\tabstention: ", abstention_turns_sum/abstention_session_sum)