import json
import pandas as pd
import json


with open('Trainingdata_dpo.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        record = json.loads(line.strip())
        print(record)



df = pd.read_parquet("hf://datasets/argilla/distilabel-math-preference-dpo/data/train-00000-of-00001-f59ecdcaca8c1de3.parquet")
import pandas as pd
import json

# 'metadata', 'instruction', 'chosen_response', 'chosen_rating', 'rejected_response', 'rejected_rating'

def convert_row(row):
    return {
        'messages': [
            {'role': 'user', 'content': row['instruction']}
        ],
        'chosen': {
            'role': 'assistant',
            'content': row['chosen_response']
        },
        'rejected': {
            'role': 'assistant',
            'content': row['rejected_response']
        }
    }


output_file = 'output.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        new_entry = convert_row(row)
        f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

print(f"jsonl file has been {output_file}")
