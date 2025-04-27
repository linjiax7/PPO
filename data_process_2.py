import pandas as pd
import json

splits = {'train': 'python/train-00000-of-00001.parquet', 'validation': 'python/validation-00000-of-00001.parquet', 'test': 'python/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/stojchet/dpo-deepseek-coder-1.3b-base-empty-fn/" + splits["train"])


def convert_row(row):
    return {
        "messages": [
            {"role": "user", "content": row["prompt"]}
        ],
        "chosen": {"role": "assistant", "content": row["chosen"]},
        "rejected": {"role": "assistant", "content": row["rejected"]}
    }

converted_data = df.apply(convert_row, axis=1).tolist()

output_file = "output.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in converted_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"data process2 {output_file}")