#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2 -> v4 semi-automatic tagger (REVIEW mode)
- symptoms (0~2) + behaviors (0~2) + severity
- Writes incrementally; supports resume by skipping existing ids in output.

Usage:
  python tagger_v4_review.py --in data_v2/train_v2.jsonl --out data_v4/train_v4.jsonl --split train --review
  python tagger_v4_review.py --in data_v2/valid_v2.jsonl --out data_v4/valid_v4.jsonl --split valid --review
"""

import argparse
import json
import os
import re
from typing import Dict, List, Set, Tuple


def _re(*patterns: str) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns]


# -------------------------
# v4 symptom tags
# -------------------------
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
    "햇빛/실내",
]

# Symptom rules (use your tuned version; includes key patches)
TAG_RULES: Dict[str, List[re.Pattern]] = {
    "피로": _re(
        r"피곤", r"피로", r"기운(이)? 없", r"무기력", r"지치", r"컨디션(이)? 안",
        r"회복(이)? (쉽게 )?안", r"회복(이)? 느리",
        r"체력.*(떨어|부족|약)",
        r"활력(이)? 부족"  # ✅ from your last remaining blank
    ),
    "수면": _re(
        r"수면", r"잠(이)? 안", r"잠들기", r"잠들기까지", r"자주 깨", r"뒤척", r"불면",
        r"숙면", r"개운(하지)? 않", r"새벽",
        r"잠(이)? 얕", r"깨(요|서|는)"
    ),
    "스트레스": _re(
        r"스트레스", r"긴장", r"예민", r"불안", r"초조", r"압박", r"화가", r"짜증"
    ),
    "면역": _re(
        r"면역", r"면역력",
        r"감기", r"몸살", r"자주 아프", r"병치레", r"컨디션(이)? 자주 무너",
        r"구내염", r"입 ?안(에)? (염증|헐)"
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
        r"뭉쳐", r"뭉치"
    ),
    "빈혈/어지럼": _re(
        r"빈혈", r"어지럽", r"현기증", r"핑(이)? 돌", r"창백", r"눈앞(이)? 깜깜",
        r"숨(이)? 차", r"호흡(이)? (가쁘|힘들)"
    ),
    "집중/인지": _re(
        r"집중력", r"집중(이)? .*안 되", r"멍(하)?",
        r"기억력", r"기억력.*(떨어|저하|감소)",
        r"기억(이)? 안", r"건망증", r"머리(가)? (안|잘) 돌아", r"인지"
    ),
    "피부/손톱/머리": _re(
        r"피부", r"피부.*트러블",
        r"여드름", r"각질", r"건조",
        r"손톱", r"머리카락", r"탈모"
    ),
    "간/회식": _re(
        r"간(이)? (피곤|안 좋|걱정)", r"숙취", r"회식", r"과음",
        r"술", r"음주", r"마시(는)? 편", r"자주 마시"
    ),
    "햇빛/실내": _re(
        r"실내", r"햇빛", r"자외선",
        r"밖에 나가(는|는)? 시간", r"밖(에)? (잘 )?안(나가|나가요|나감)", r"야외(활동)?", r"집(에)?만",
        r"\d+분도 안"
    ),
}


# -------------------------
# v4 behavior codes
# -------------------------
BEHAVIORS = [
    "오메가3_섭취부족",
    "햇빛노출부족",
    "음주빈도높음",
    "고지방식",
    "불규칙식사",
    "패스트푸드빈도높음",
]

BEHAVIOR_RULES: Dict[str, List[re.Pattern]] = {
    "오메가3_섭취부족": _re(
        r"(등푸른)? ?생선.*(거의|잘|전혀)? ?(안|없)",
        r"생선.*섭취.*(거의|없|안)",
        r"오메가 ?-?3.*(섭취|먹).*?(거의|없|안)"
    ),
    "햇빛노출부족": _re(
        r"실내", r"햇빛.*(못|거의|잘)? ?(쬐|보)", r"밖에 나가(는|는)? 시간", r"\d+분도 안"
    ),
    "음주빈도높음": _re(
        r"술.*(자주|거의 매일|잦)", r"음주.*(자주|잦)", r"과음", r"회식.*(잦|자주)"
    ),
    "고지방식": _re(
        r"기름진 음식.*(자주|많)", r"튀김.*(자주|많)", r"지방.*(많|과다)"
    ),
    "불규칙식사": _re(
        r"식사.*(불규칙|거르)", r"끼니.*(불규칙|거르|거름)"
    ),
    "패스트푸드빈도높음": _re(
        r"패스트푸드.*(자주|많)", r"인스턴트.*(자주|많)"
    ),
}


# Severity heuristic
SEVERITY_STRONG = _re(r"매우", r"심하", r"너무", r"자주", r"계속", r"항상", r"거의 매일", r"극심")
SEVERITY_MILD = _re(r"가끔", r"조금", r"약간", r"종종")


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def write_jsonl_append(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_out_ids(out_path: str) -> Set[str]:
    if not os.path.exists(out_path):
        return set()
    ids = set()
    for _, obj in read_jsonl(out_path):
        rid = obj.get("id")
        if isinstance(rid, str) and rid:
            ids.add(rid)
    return ids


def infer_severity(text: str) -> int:
    t = text.strip()
    if any(p.search(t) for p in SEVERITY_STRONG):
        return 3
    if any(p.search(t) for p in SEVERITY_MILD):
        return 1
    return 2


def propose_from_rules(text: str, rules: Dict[str, List[re.Pattern]], allowed: List[str], cap: int) -> List[str]:
    t = text.strip()
    hits: List[Tuple[str, int]] = []
    for key, patterns in rules.items():
        score = 0
        for p in patterns:
            if p.search(t):
                score += 1
        if score > 0:
            hits.append((key, score))

    hits.sort(key=lambda x: (-x[1], allowed.index(x[0])))
    tags = [k for k, _ in hits]

    # keep conservative
    return tags[:cap]


def parse_manual_list(s: str, allowed: List[str], cap: int) -> List[str]:
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    bad = [p for p in parts if p not in allowed]
    if bad:
        raise ValueError(f"Unknown item(s): {bad}. Allowed: {allowed}")
    return parts[:cap]


def review_loop(in_path: str, out_path: str, split: str):
    done_ids = load_existing_out_ids(out_path)

    print(f"[INFO] Input : {in_path}")
    print(f"[INFO] Output: {out_path}")
    if done_ids:
        print(f"[INFO] Resume: {len(done_ids)} already tagged; will skip.\n")

    print("[HOW TO REVIEW]")
    print("  Enter : accept proposed symptoms+behaviors")
    print("  es    : edit symptoms (comma; max2)")
    print("  eb    : edit behaviors (comma; max2)")
    print("  n     : set both empty")
    print("  s     : skip")
    print("  q     : quit\n")

    written = 0
    skipped = 0

    for line_no, obj in read_jsonl(in_path):
        rid = obj.get("id")
        text = (obj.get("text") or "").strip()
        if not rid or not text:
            raise ValueError(f"Missing 'id' or 'text' at {in_path}:{line_no}")

        if rid in done_ids:
            continue

        proposed_sym = propose_from_rules(text, TAG_RULES, SYMPTOMS, cap=2)
        proposed_beh = propose_from_rules(text, BEHAVIOR_RULES, BEHAVIORS, cap=2)
        sev = infer_severity(text)

        print("------------------------------------------------------------")
        print(f"{split.upper()} | line {line_no} | id={rid}")
        print(f"TEXT: {text}")
        print(f"LABEL: {obj.get('label_ko')} (id={obj.get('label_id')})")
        print(f"PROPOSED symptoms : {proposed_sym}")
        print(f"PROPOSED behaviors: {proposed_beh}")
        print(f"severity: {sev}")
        print(f"Allowed symptoms : {SYMPTOMS}")
        print(f"Allowed behaviors: {BEHAVIORS}")

        cmd = input("Action [Enter/es/eb/n/s/q]: ").strip().lower()

        if cmd == "q":
            print("[INFO] Quit.")
            break
        if cmd == "s":
            skipped += 1
            continue

        sym = proposed_sym
        beh = proposed_beh

        if cmd == "es":
            manual = input("Enter symptoms (comma; max2) | empty=none: ").strip()
            try:
                sym = parse_manual_list(manual, SYMPTOMS, cap=2)
            except ValueError as ve:
                print(f"[WARN] {ve} -> using proposed symptoms.")
                sym = proposed_sym
        elif cmd == "eb":
            manual = input("Enter behaviors (comma; max2) | empty=none: ").strip()
            try:
                beh = parse_manual_list(manual, BEHAVIORS, cap=2)
            except ValueError as ve:
                print(f"[WARN] {ve} -> using proposed behaviors.")
                beh = proposed_beh
        elif cmd == "n":
            sym, beh = [], []

        rec = dict(obj)
        rec["features"] = {
            "symptoms": sym,
            "behaviors": beh,
            "severity": sev
        }

        write_jsonl_append(out_path, rec)
        written += 1

    print("\n[SUMMARY]")
    print(f"  written: {written}")
    print(f"  skipped: {skipped}")
    print(f"  output : {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--split", required=True, choices=["train", "valid"])
    ap.add_argument("--review", action="store_true")
    args = ap.parse_args()

    if not args.review:
        raise SystemExit("Use --review for interactive tagging.")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    review_loop(args.in_path, args.out_path, args.split)


if __name__ == "__main__":
    main()
