import json

def load_kb(path="kb.json"):
    with open(path, "r") as f:
        return {item["question"]: item["answer"] for item in json.load(f)}

def validate_answer(question, answer, kb):
    if question in kb:
        if answer.strip().lower() == kb[question].strip().lower():
            return "CORRECT"
        else:
            return "RETRY: answer differs from KB"
    else:
        return "RETRY: out-of-domain"
