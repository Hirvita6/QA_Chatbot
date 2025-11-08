import os
import json
from datetime import datetime
from typing import List
from transformers import pipeline

flan_model = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
roberta_model = pipeline("question-answering", model="deepset/tinyroberta-squad2", device=-1)

# ------------------------------
# Logging function
# ------------------------------
def log_qa_interaction(question, answer, score=None, log_path="logs/qa_interactions.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat() + "Z",
        "question": question,
        "answer": answer,
    }
    if score is not None:
        entry["Confidence score"] = score
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ------------------------------
# FLAN-T5 version (Generative)
# ------------------------------
def answer_with_flan_t5(question: str, contexts: List[str]) -> str:
    context_text = "\n\n".join(contexts)
    prompt = f"""You are a helpful and precise AI assistant.
Answer ONLY based on the context below.
If the answer is not in the context, respond with: "not relevant to the context".

Context:
{context_text}

Question: {question}
"""
    result = flan_model(prompt, max_length=150, min_length=30, do_sample=False)[0]
    answer = result.get("generated_text", "").strip()

    if not answer or "not relevant" in answer.lower() or "i don't know" in answer.lower():
        answer = "not relevant to the context"

    log_qa_interaction(question, answer)
    return answer


# ------------------------------
# RoBERTa version (Extractive)
# ------------------------------
def answer_with_roberta(question: str, contexts: List[str]) -> str:
    context_text = "\n".join(contexts)
    result = roberta_model(question=question, context=context_text)
    answer = result.get("answer", "").strip()
    score = result.get("score", 0)

    if not answer or score < 0.1:
        answer = "I don’t know, but I can help search for more info."

    log_qa_interaction(question, answer, score)
    return answer


