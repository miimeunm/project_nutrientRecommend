#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert legacy JSONL files ({"text": ..., "label": ...}) into v2 JSONL with
stable label_id/label_ko and minimal metadata.

Input (JSONL; one JSON object per line):
  {"text": "....", "label": "Vitamin C"}

Output (JSONL; one JSON object per line):
  {
    "id": "train_000001",
    "text": "....",
    "label_id": 8,
    "label_ko": "비타민 C",
    "source": "free_text",
    "split": "train",
    "meta": {"lang": "ko", "note": ""}
  }

Usage (Windows PowerShell):
  python convert_dataset_v2.py --train train.json --valid valid.json --outdir data_v2

Notes:
- Assumes input is JSONL even if file extension is .json.
- model.pt is unrelated; this is only data conversion.
"""

import argparse
import json
import os
from typing import Dict, Tuple


# Canonical 10-label set (fixed order). Use this everywhere (train/infer/UI).
LABELS_KO = [
    "비타민 D",      # 0
    "마그네슘",       # 1
    "오메가-3",       # 2
    "비타민 B12",     # 3
    "철분",           # 4
    "아연",           # 5
    "칼슘",           # 6
    "프로바이오틱스",  # 7
    "비타민 C",       # 8
    "밀크시슬",       # 9
]

# Map legacy English labels to canonical label_id + label_ko.
# Extend/adjust if your legacy labels differ.
LEGACY_EN_TO_ID: Dict[str, int] = {
    "Vitamin D": 0,
    "Magnesium": 1,
    "Omega-3": 2,
    "Vitamin B12": 3,
    "Iron": 4,
    "Zinc": 5,
    "Calcium": 6,
    "Probiotics": 7,
    "Vitamin C": 8,
    "Milk Thistle": 9,
    # Optional aliases (common variants)
    "Omega 3": 2,
    "Omega3": 2,
    "Vit D": 0,
    "Vit C": 8,
    "Vit B12": 3,
    "B12": 3,
}


def normalize_label(legacy_label: str) -> Tuple[int, str]:
    legacy_label = legacy_label.strip()
    if legacy_label not in LEGACY_EN_TO_ID:
        raise ValueError(
            f"Unknown legacy label: '{legacy_label}'. "
            f"Add it to LEGACY_EN_TO_ID mapping."
        )
    label_id = LEGACY_EN_TO_ID[legacy_label]
    return label_id, LABELS_KO[label_id]


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}\n{line}\n{e}") from e


def write_jsonl(path: str, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def convert_file(in_path: str, split: str, out_path: str, source: str = "free_text"):
    out = []
    idx = 0

    for line_no, obj in read_jsonl(in_path):
        text = (obj.get("text") or "").strip()
        legacy_label = (obj.get("label") or "").strip()

        if not text:
            raise ValueError(f"Missing/empty 'text' at {in_path}:{line_no}")
        if not legacy_label:
            raise ValueError(f"Missing/empty 'label' at {in_path}:{line_no}")

        label_id, label_ko = normalize_label(legacy_label)

        idx += 1
        rec = {
            "id": f"{split}_{idx:06d}",
            "text": text,
            "label_id": label_id,
            "label_ko": label_ko,
            "source": source,
            "split": split,
            "meta": {
                "lang": "ko",
                "note": "",
            },
        }
        out.append(rec)

    write_jsonl(out_path, out)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to legacy train JSONL (e.g., train.json)")
    ap.add_argument("--valid", required=True, help="Path to legacy valid JSONL (e.g., valid.json)")
    ap.add_argument("--outdir", default="data_v2", help="Output directory for v2 JSONL files")
    ap.add_argument("--source", default="free_text", help="Source tag for records (default: free_text)")
    args = ap.parse_args()

    out_train = os.path.join(args.outdir, "train_v2.jsonl")
    out_valid = os.path.join(args.outdir, "valid_v2.jsonl")

    n_train = convert_file(args.train, "train", out_train, source=args.source)
    n_valid = convert_file(args.valid, "valid", out_valid, source=args.source)

    print("Conversion complete.")
    print(f"- Train: {n_train} records -> {out_train}")
    print(f"- Valid: {n_valid} records -> {out_valid}")
    print("Label set (id -> ko):")
    for i, ko in enumerate(LABELS_KO):
        print(f"  {i}: {ko}")


if __name__ == "__main__":
    main()
