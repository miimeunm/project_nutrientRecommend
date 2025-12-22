import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import NutrientClassifier
from survey_questions import SURVEY_QUESTIONS
from build_input import build_model_input


# ======================
# 설정
# ======================
MODEL_PATH = "model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    0: "비타민 D",
    1: "마그네슘",
    2: "오메가-3",
    3: "비타민 B12",
    4: "철분",
    5: "아연",
    6: "칼슘",
    7: "프로바이오틱스",
    8: "비타민 C",
    9: "밀크시슬",
}


# ======================
# 설문 입력 함수
# ======================
def run_survey():
    responses = {}

    print("\n📝 건강 설문을 시작합니다 (1~5 숫자 입력)\n")

    for idx, q in enumerate(SURVEY_QUESTIONS, 1):
        print(f"\nQ{idx}. {q['question']}")
        for i, option in enumerate(q["options"], 1):
            print(f"  {i}. {option}")

        while True:
            try:
                choice = int(input("선택: "))
                if 1 <= choice <= 5:
                    responses[q["key"]] = choice
                    break
                else:
                    print("⚠️ 1~5 사이 숫자를 입력해주세요.")
            except ValueError:
                print("⚠️ 숫자를 입력해주세요.")

    return responses


# ======================
# 모델 로드
# ======================
tokenizer = AutoTokenizer.from_pretrained(
    "monologg/kobert",
    trust_remote_code=True
)

model = NutrientClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# ======================
# 실행부 (엔트리 포인트)
# ======================
if __name__ == "__main__":

    print("\n🥗 영양소 추천 설문 모델\n")

    # 1️⃣ 설문 진행
    responses = run_survey()

    # 2️⃣ 설문 → 문장 변환
    input_text = build_model_input(responses)

    print("\n📄 모델 입력 문장:")
    print(input_text)

    # 3️⃣ 모델 추론
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        probs = F.softmax(logits, dim=1).squeeze()

    # 4️⃣ TOP-3 출력
    topk = torch.topk(probs, k=3)

    print("\n👉 추천 영양소 TOP 3:")
    for rank, (idx, score) in enumerate(zip(topk.indices, topk.values), start=1):
        label = LABEL_MAP[idx.item()]
        print(f"{rank}️⃣ {label:<10} (신뢰도 {score.item():.2f})")
