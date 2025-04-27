import os
import json
import requests
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm


API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def query_model(messages):

    try:
        response = requests.post(
            url=BASE_URL + "/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": "qwen-plus", "messages": messages, "temperature": 0.0}
        )
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"API Error: {e}")
        return None

def compute_bleu(reference, hypothesis):
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    smoothing = SmoothingFunction().method1  # 避免 BLEU 0 分
    return sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing)

def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

def load_jsonl(file_path, sample_size=20):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break  
            data.append(json.loads(line))
    return data


def evaluate_model(jsonl_file, test_size):
    data = load_jsonl(jsonl_file, sample_size= test_size) 

    total_bleu = 0
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0
    count = 0

    for sample in tqdm(data, desc="Evaluating"):
        messages = sample["messages"]
        reference_answer = sample["chosen"]["content"]
        model_response = query_model(messages)
        if model_response is None:
            continue

        bleu_score = compute_bleu(reference_answer, model_response)

        rouge_scores = compute_rouge(reference_answer, model_response)

        total_bleu += bleu_score
        total_rouge1 += rouge_scores["rouge1"].fmeasure
        total_rouge2 += rouge_scores["rouge2"].fmeasure
        total_rougeL += rouge_scores["rougeL"].fmeasure
        count += 1

    avg_bleu = total_bleu / count
    avg_rouge1 = total_rouge1 / count
    avg_rouge2 = total_rouge2 / count
    avg_rougeL = total_rougeL / count

    print(f"\nFinal Evaluation Results:")
    print(f"BLEU Score: {avg_bleu:.4f}")
    print(f"ROUGE-1 Score: {avg_rouge1:.4f}")
    print(f"ROUGE-2 Score: {avg_rouge2:.4f}")
    print(f"ROUGE-L Score: {avg_rougeL:.4f}")


jsonl_file = "Human-Like-DPO.jsonl"
test_size = 20
evaluate_model(jsonl_file, test_size)


