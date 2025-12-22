import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

LABEL2ID = {
    "Vitamin D": 0,
    "Magnesium": 1,
    "Omega-3": 2,
    "Vitamin B12": 3,
    "Iron": 4,
    "Zinc": 5,
    "Calcium": 6,
    "Probiotics": 7,
    "Vitamin C": 8,
    "Milk Thistle": 9
}

tokenizer = AutoTokenizer.from_pretrained(
    "monologg/kobert",
    trust_remote_code=True
)

class NutrientDataset(Dataset):
    def __init__(self, path, max_len=64):
        self.data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(LABEL2ID[item["label"]])
        }
