import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)

    return acc, f1, cm
