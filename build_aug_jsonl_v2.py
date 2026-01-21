import argparse
import json
import os


# v4.1 features -> 자연어 힌트(학습 입력용)
SYMPTOM_KO = {
    "피로": "피로/무기력",
    "수면": "수면 문제",
    "스트레스": "스트레스",
    "면역": "면역 저하/감기 잦음",
    "소화/속불편": "소화 불편",
    "장건강": "장 건강/배변 문제",
    "뼈/관절": "뼈/관절",
    "근육/쥐": "근육/경련",
    "빈혈/어지럼": "빈혈/어지럼",
    "집중/인지": "집중/기억",
    "피부/손톱/머리": "피부/손톱/모발",
    "간/회식": "간 피로/음주 영향",
    "햇빛/실내": "햇빛 노출 부족",
}

BEHAVIOR_KO = {
    "오메가3_섭취부족": "등푸른 생선/오메가-3 섭취 부족",
    "햇빛노출부족": "햇빛 노출 부족",
    "음주빈도높음": "음주/회식 빈도 높음",
    "고지방식": "기름진 음식/고지방식",
    "불규칙식사": "불규칙한 식사",
    "패스트푸드빈도높음": "패스트푸드/인스턴트 섭취 많음",
    "과일섭취부족": "과일 섭취 부족",
    "유제품섭취부족": "유제품 섭취 부족",
}


# label_id / label_ko -> dataset.py가 요구하는 영문 label 문자열로 매핑
LABEL_ID_TO_EN = {
    0: "Vitamin D",
    1: "Magnesium",
    2: "Omega-3",
    3: "Vitamin B12",
    4: "Iron",
    5: "Zinc",
    6: "Calcium",
    7: "Probiotics",
    8: "Vitamin C",
    9: "Milk Thistle",
}

LABEL_KO_TO_EN = {
    "비타민 D": "Vitamin D",
    "마그네슘": "Magnesium",
    "오메가-3": "Omega-3",
    "비타민 B12": "Vitamin B12",
    "철분": "Iron",
    "아연": "Zinc",
    "칼슘": "Calcium",
    "프로바이오틱스": "Probiotics",
    "비타민 C": "Vitamin C",
    "밀크시슬": "Milk Thistle",
}


def render_features(features: dict) -> str:
    if not isinstance(features, dict):
        return ""
    symptoms = features.get("symptoms") or []
    behaviors = features.get("behaviors") or []

    sym_text = ", ".join([SYMPTOM_KO.get(s, s) for s in symptoms]) if symptoms else ""
    beh_text = ", ".join([BEHAVIOR_KO.get(b, b) for b in behaviors]) if behaviors else ""

    parts = []
    if sym_text:
        parts.append(f"관찰된 증상: {sym_text}.")
    if beh_text:
        parts.append(f"관찰된 생활/섭취: {beh_text}.")
    return " ".join(parts)


def to_label_en(obj: dict) -> str:
    # Priority: existing label (already English) > label_id > label_ko
    if isinstance(obj.get("label"), str) and obj["label"].strip():
        return obj["label"].strip()

    if "label_id" in obj:
        try:
            lid = int(obj["label_id"])
            if lid in LABEL_ID_TO_EN:
                return LABEL_ID_TO_EN[lid]
        except Exception:
            pass

    if isinstance(obj.get("label_ko"), str):
        ko = obj["label_ko"].strip()
        if ko in LABEL_KO_TO_EN:
            return LABEL_KO_TO_EN[ko]

    raise ValueError(f"Cannot determine label in object id={obj.get('id')}. "
                     f"Expected one of: label (en), label_id, label_ko.")


def convert(in_path: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = (obj.get("text") or "").strip()
            if not text:
                raise ValueError(f"Missing text at {in_path}:{line_no}")

            feats = render_features(obj.get("features", {}))
            text_aug = f"{feats} 원문: {text}".strip() if feats else text

            label_en = to_label_en(obj)

            # Dataset이 기대하는 최소 필드만 맞춰서 출력(불필요 필드 유지해도 무방)
            obj["text"] = text_aug
            obj["label"] = label_en  # ✅ 핵심: dataset.py 호환

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] wrote {n} lines -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()
    convert(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
