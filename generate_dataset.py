# generate_dataset.py
import json
import random
from seed_sentences import SEED_SENTENCES

OUTPUT_TRAIN = "train.jsonl"
OUTPUT_VALID = "valid.jsonl"

TRAIN_RATIO = 0.8          # train / valid 비율
AUGMENT_TIMES = 5          # 문장당 증강 횟수
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# 간단한 말투/강조 증강용 prefix/suffix
PREFIXES = [
    "",
    "요즘 ",
    "최근에 ",
    "전반적으로 ",
]

SUFFIXES = [
    "",
    " 그런 편이에요.",
    " 느낌이에요.",
    " 같아요.",
]

def augment_sentence(sentence):
    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIXES)

    s = sentence.rstrip(".")

    return f"{prefix}{s}{suffix}".strip()


def generate_samples():
    samples = []

    for label, sentences in SEED_SENTENCES.items():
        for sent in sentences:
            # 원본 문장
            samples.append({
                "text": sent,
                "label": label
            })

            # 증강 문장
            for _ in range(AUGMENT_TIMES):
                aug = augment_sentence(sent)
                samples.append({
                    "text": aug,
                    "label": label
                })

    random.shuffle(samples)
    return samples


def split_and_save(samples):
    split_idx = int(len(samples) * TRAIN_RATIO)
    train_samples = samples[:split_idx]
    valid_samples = samples[split_idx:]

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(OUTPUT_VALID, "w", encoding="utf-8") as f:
        for s in valid_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Train samples: {len(train_samples)}")
    print(f"Valid samples: {len(valid_samples)}")


if __name__ == "__main__":
    samples = generate_samples()
    split_and_save(samples)
