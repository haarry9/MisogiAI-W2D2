from transformers import pipeline
import json
import time
from validator import load_kb, validate_answer

# Load the model
llm = pipeline("text-generation", model="gpt2")  # Light model, no GPU needed
kb = load_kb()

# Questions
known_qs = list(kb.keys())
new_qs = [
    "What is the speed of light in vacuum?",
    "Who discovered penicillin?",
    "What is the population of Mars?",
    "What is the square root of -1?",
    "What is the largest mammal in the ocean?"
]

all_qs = known_qs + new_qs

# Run log
log_lines = []

# Track summary
summary = {
    "total": 0,
    "correct": 0,
    "retry_success": 0,
    "retry_fail": 0,
    "mismatch": 0,
    "out_of_domain": 0
}

def extract_answer(response):
    return response[0]["generated_text"].split("?")[-1].strip().split(".")[0]

for q in all_qs:
    summary["total"] += 1
    response = llm(q, max_new_tokens=20)
    ans1 = extract_answer(response)

    status = validate_answer(q, ans1, kb)
    log_lines.append(f"Q: {q}\nA1: {ans1}\n→ {status}")

    if status == "CORRECT":
        summary["correct"] += 1
    elif "RETRY" in status:
        retry_response = llm(q, max_new_tokens=20)
        ans2 = extract_answer(retry_response)
        retry_status = validate_answer(q, ans2, kb)
        log_lines.append(f"A2: {ans2}\n→ {retry_status}\n")

        if retry_status == "CORRECT":
            summary["retry_success"] += 1
        else:
            summary["retry_fail"] += 1

        if status == "RETRY: out-of-domain":
            summary["out_of_domain"] += 1
        else:
            summary["mismatch"] += 1

    log_lines.append("-" * 50)

# Save logs
with open("run.log", "w") as f:
    f.write("\n".join(log_lines))

# Save summary
with open("summary.md", "w") as f:
    f.write("## Run Summary\n\n")
    for k, v in summary.items():
        f.write(f"- **{k}**: {v}\n")

print("✅ Completed! Logs saved to run.log, summary saved to summary.md.")
