import torch.nn as nn
from transformers import BertModel

NUM_LABELS = 10

class NutrientClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.fc = nn.Linear(768, NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0]
        return self.fc(cls)
