import pandas as pd
import json


df = pd.read_json("hf://datasets/HumanLLMs/Human-Like-DPO-Dataset/data.json")

def convert_row(row):
    return {
        "messages": [
            {"role": "user", "content": row["prompt"]}
        ],
        "chosen": {"role": "assistant", "content": row["chosen"]},
        "rejected": {"role": "assistant", "content": row["rejected"]}
    }

converted_data = df.apply(convert_row, axis=1).tolist()

output_file = "Human-Like-DPO.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in converted_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"data process3 {output_file}")