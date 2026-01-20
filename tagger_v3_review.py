#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2 -> v3 (features.symptoms) semi-automatic tagger (REVIEW mode)

Input (v2 JSONL):
  {"id":"train_000001","text":"...","label_id":8,"label_ko":"비타민 C", ...}

Output (v3 JSONL):
  ... + "features": {"symptoms": ["면역", ...], "severity": 2}

REVIEW mode:
- For each sample, script proposes symptom tags based on keyword rules.
- You can accept as-is, edit, skip, or quit.
- Writes output incrementally so you don't lose work.

Usage (PowerShell):
  python tagger_v3_review.py --in data_v2/train_v2.jsonl --out data_v3/train_v3.jsonl --split train --review

  python tagger_v3_review.py --in data_v2/valid_v2.jsonl --out data_v3/valid_v3.jsonl --split valid --review

Tip:
- Start with train first, tag ~30-50 samples, then refine keyword rules.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Set, Tuple


# v3 symptom tag set (fixed strings; keep consistent)
SYMPTOMS = [
    "피로",
    "수면",
    "스트레스",
    "면역",
    "소화/속불편",
    "장건강",
    "뼈/관절",
    "근육/쥐",
    "빈혈/어지럼",
    "집중/인지",
    "피부/손톱/머리",
    "간/회식",
    "햇빛/실내",   #신규
]



def _re(*patterns: str) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns]


# Keyword/pattern rules for proposing tags.
# Keep these conservative to avoid over-tagging.
TAG_RULES: Dict[str, List[re.Pattern]] = {
    "피로": _re(
        r"피곤", r"피로", r"기운(이)? 없", r"무기력", r"지치", r"컨디션(이)? 안",
        r"회복(이)? (쉽게 )?안", r"회복(이)? 느리",
        r"체력(이)? (떨어|없|부족|약)",
        r"체력.*(떨어|부족|약)" # 보강
    ),

    "수면": _re(
        r"수면", r"잠(이)? 안", r"잠들기", r"잠들기까지", r"자주 깨", r"뒤척", r"불면",
        r"숙면", r"개운(하지)? 않", r"새벽", r"잠(이)? 얕", r"깨(는|요|서)"
    ),

    "스트레스": _re(
        r"스트레스", r"긴장", r"예민", r"불안", r"초조", r"압박", r"화가", r"짜증"
    ),

    "면역": _re(
        r"면역", r"면역력", r"감기",  # 보강
        r"감기(가)? (자주|잦)", r"몸살", r"자주 아프", r"병치레", r"컨디션(이)? 자주 무너",
        r"구내염", r"입 ?안(에)? (염증|헐)"  # '입 안 염증' 커버
    ),

    "소화/속불편": _re(
        r"소화(가)? 안", r"더부룩", r"속(이)? (불편|쓰리|답답)", r"체한", r"메스껍", r"구역"
    ),

    "장건강": _re(
        r"장(이)? 안", r"장 ?트러블", r"배변", r"변비", r"설사", r"가스(가)? (차|많)", r"복부팽만"
    ),

    "뼈/관절": _re(
        r"뼈", r"관절", r"골다공", r"골밀도"
    ),

    "근육/쥐": _re(
        r"쥐(가)? 나", r"경련",
        r"근육(이)? (뭉치|뻐근|당기)", r"근육통",
        r"뭉쳐", r"뭉치"  # 보강(누락 방지)
    ),

    "빈혈/어지럼": _re(
        r"빈혈", r"어지럽", r"현기증", r"핑(이)? 돌", r"창백", r"눈앞(이)? 깜깜",
        r"숨(이)? 차", r"호흡(이)? (가쁘|힘들)"  # 보강
    ),

    "집중/인지": _re(
        r"집중(이)? 안", r"멍(하)?", r"기억(이)? 안", r"건망증", r"머리(가)? (안|잘) 돌아", r"인지", r"기억력", r"기억력(이)? (떨어|감소|저하)", r"집중력", r"집중력(이)? (떨어|저하)"
    ),

    # 오탐 방지: '트러블' 단독 키워드 제거
    # - '장 트러블' 때문에 피부로 잘못 매칭되던 문제 해결
    "피부/손톱/머리": _re(
        r"피부", r"피부.*트러블",  # ✅ 피부+트러블 조합만
        r"여드름", r"각질", r"건조",
        r"손톱", r"머리카락", r"탈모"
    ),

    "간/회식": _re(
        r"간(이)? (피곤|안 좋|걱정)", r"숙취", r"회식", r"과음",
        r"술", r"음주", r"마시(는)? 편", r"자주 마시"  # ✅ 보강
    ),

    # 신규: 햇빛/실내 (비타민 D 케이스 커버)
    "햇빛/실내": _re(
        r"실내", r"햇빛", r"자외선", r"밖에 나가(는|는)? 시간", r"햇빛(을)? (쬐|보)",
        r"밖(에)? (잘 )?안(나가|나가요|나감)", r"야외(활동)?", r"집(에)?만"
    ),
}



# Severity heuristic: 1 (mild) / 2 (default) / 3 (strong)
SEVERITY_STRONG = _re(r"매우", r"심하", r"너무", r"자주", r"계속", r"항상", r"거의 매일", r"극심")
SEVERITY_MILD = _re(r"가끔", r"조금", r"약간", r"종종")


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


def write_jsonl_append(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def propose_tags(text: str) -> List[str]:
    t = text.strip()
    hits: List[Tuple[str, int]] = []
    for tag, patterns in TAG_RULES.items():
        score = 0
        for p in patterns:
            if p.search(t):
                score += 1
        if score > 0:
            hits.append((tag, score))

    # Sort by score desc, then stable order
    hits.sort(key=lambda x: (-x[1], SYMPTOMS.index(x[0])))

    # Conservative cap: at most 2 tags (per earlier rule)
    tags = [tag for tag, _ in hits]

    # 햇빛/실내는 있으면 우선 포함 (강한 신호)
    if "햇빛/실내" in tags:
        tags.remove("햇빛/실내")
        tags = ["햇빛/실내"] + tags

    # Conservative cap: at most 2 tags
    return tags[:2]


def infer_severity(text: str) -> int:
    t = text.strip()
    if any(p.search(t) for p in SEVERITY_STRONG):
        return 3
    if any(p.search(t) for p in SEVERITY_MILD):
        return 1
    return 2


def parse_manual_tags(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    # allow comma-separated
    parts = [p.strip() for p in s.split(",") if p.strip()]
    # validate
    bad = [p for p in parts if p not in SYMPTOMS]
    if bad:
        raise ValueError(f"Unknown tag(s): {bad}. Allowed: {SYMPTOMS}")
    # cap to 2 tags (enforce rule)
    return parts[:2]


def load_existing_out_ids(out_path: str) -> Set[str]:
    if not os.path.exists(out_path):
        return set()
    ids = set()
    for _, obj in read_jsonl(out_path):
        rid = obj.get("id")
        if isinstance(rid, str) and rid:
            ids.add(rid)
    return ids


def review_loop(in_path: str, out_path: str, split: str):
    done_ids = load_existing_out_ids(out_path)

    print(f"[INFO] Input : {in_path}")
    print(f"[INFO] Output: {out_path}")
    if done_ids:
        print(f"[INFO] Resume mode: {len(done_ids)} records already tagged in output; they will be skipped.")

    print("\n[HOW TO REVIEW]")
    print("  Enter : accept proposed tags")
    print("  e     : edit tags manually (comma-separated; max 2)")
    print("  n     : no tags (symptoms = [])")
    print("  s     : skip this sample (do not write)")
    print("  q     : quit immediately\n")

    count_written = 0
    count_skipped = 0

    for line_no, obj in read_jsonl(in_path):
        rid = obj.get("id")
        text = (obj.get("text") or "").strip()

        if not rid or not text:
            raise ValueError(f"Missing 'id' or 'text' at {in_path}:{line_no}")

        if rid in done_ids:
            continue

        proposed = propose_tags(text)
        sev = infer_severity(text)

        print("------------------------------------------------------------")
        print(f"{split.upper()} | line {line_no} | id={rid}")
        print(f"TEXT: {text}")
        print(f"LABEL: {obj.get('label_ko')} (id={obj.get('label_id')})")
        print(f"PROPOSED symptoms: {proposed} | severity: {sev}")
        print(f"Allowed tags: {SYMPTOMS}")

        cmd = input("Action [Enter/e/n/s/q]: ").strip().lower()

        if cmd == "q":
            print("[INFO] Quit requested. Exiting.")
            break
        if cmd == "s":
            count_skipped += 1
            continue

        if cmd == "e":
            manual = input("Enter tags (comma-separated; max 2). Example: 피로,면역  | empty=none: ").strip()
            try:
                tags = parse_manual_tags(manual)
            except ValueError as ve:
                print(f"[WARN] {ve}  -> using proposed tags instead.")
                tags = proposed
        elif cmd == "n":
            tags = []
        else:
            tags = proposed

        rec = dict(obj)  # copy v2 record
        rec["features"] = {
            "symptoms": tags,
            "severity": sev
        }

        write_jsonl_append(out_path, rec)
        count_written += 1

    print("\n[SUMMARY]")
    print(f"  written: {count_written}")
    print(f"  skipped: {count_skipped}")
    print(f"  output : {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input v2 JSONL path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output v3 JSONL path")
    ap.add_argument("--split", required=True, choices=["train", "valid"], help="Split name")
    ap.add_argument("--review", action="store_true", help="Run interactive review mode")
    args = ap.parse_args()

    if not args.review:
        raise SystemExit("This script currently supports only --review mode. (You can ask me for --auto mode.)")

    # safety: prevent accidental overwrite without intent
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    review_loop(args.in_path, args.out_path, args.split)


if __name__ == "__main__":
    main()
