
def get_target(expected_sessions: list, retrieved_sessions: list):
    "Get the traget condition of one retrieval."
    expected_set = set(expected_sessions)
    retrieved_set = set(retrieved_sessions)
    intersection = expected_set & retrieved_set
    if not intersection:
        return "no-target"
    elif intersection == expected_set:
        return "fully-target"
    else:
        return "partly-target"