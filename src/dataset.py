import json
from datasets import Dataset
from src.config import LABEL2ID
from src.preprocess import clean_email

def load_hf_dataset(split: str) -> Dataset:
    records = []
    with open(f"data/processed/{split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            records.append({"text": clean_email(obj["text"]), "label": LABEL2ID[obj["label"]]})
    return Dataset.from_list(records)