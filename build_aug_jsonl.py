import argparse
import json
import os


# 사람이 읽을 수 있는 한국어 매핑(학습 입력용)
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


def render_features(features: dict) -> str:
    if not isinstance(features, dict):
        return ""

    symptoms = features.get("symptoms") or []
    behaviors = features.get("behaviors") or []

    sym_text = ", ".join([SYMPTOM_KO.get(s, s) for s in symptoms]) if symptoms else ""
    beh_text = ", ".join([BEHAVIOR_KO.get(b, b) for b in behaviors]) if behaviors else ""

    parts = []
    # “근거:” 형태로 덧붙여서 모델이 힌트로 쓰게 함
    if sym_text:
        parts.append(f"관찰된 증상: {sym_text}.")
    if beh_text:
        parts.append(f"관찰된 생활/섭취: {beh_text}.")
    return " ".join(parts)


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
            if feats:
                text_aug = f"{text} {feats}"
            else:
                text_aug = text

            # NutrientDataset이 기대하는 최소 필드: text, label (또는 label_id)
            # 당신 기존 jsonl이 label을 어떻게 쓰는지에 따라 둘 중 하나만 맞추면 됩니다.
            # 여기서는 현재 구조( label_id, label_ko )를 유지하고 text만 덮어씁니다.
            obj["text"] = text_aug

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