"Some functions to process retrieved chunks."
from utils.dates import compare_dates, date2datetime

def integrate_same_sessions(top_k_sessions_ids:list, num = 5):
    integrated_session_ids = []
    for session_id in top_k_sessions_ids:
        if len(integrated_session_ids) >= num: # 避免过多
            return integrated_session_ids
        if session_id not in integrated_session_ids:
            integrated_session_ids.append(session_id)
    return integrated_session_ids


def reorganize_evidence_sessions(evidence_sessions:list):
    return sorted(evidence_sessions, key=lambda session: date2datetime(session.session_date))


def session2context(evidence_sessions:list, process_type="raw"):
    """
    Turn a sorted evidence session list into a str.
    Args:
        evidence_sessions (list): a sorted evidence session list
        type (str): The way we process evidence sessions, namely `raw`, `compress`
    """
    context = ""
    if process_type == 'raw':
        for idx, session in enumerate(evidence_sessions):
            context += f"Session {idx}:\n" + f"Session date: {session.session_date}\n" # 先不转化为datetime格式，保留周几信息
            context += session.get_session_str()
            context += "-"*60 + '\n'
        return context
    elif process_type == 'compress':
        ...
    else:
        raise ValueError(f"Process type {process_type} is not supported.")