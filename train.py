import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from dataset import NutrientDataset
from model import NutrientClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = NutrientDataset("train.jsonl")
valid_ds = NutrientDataset("valid.jsonl")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=16)

model = NutrientClassifier().to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

def train_epoch():
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device)
        )
        loss = criterion(logits, batch["label"].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

from evaluate import evaluate

EPOCHS = 3
for epoch in range(EPOCHS):
    loss = train_epoch()
    acc, f1, cm = evaluate(model, valid_loader, device)
    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {loss:.4f}")
    print(f"Valid Acc : {acc:.4f}")
    print(f"Valid F1  : {f1:.4f}")

torch.save(model.state_dict(), "model.pt")
