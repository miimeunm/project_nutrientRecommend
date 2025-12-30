import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from model import NutrientClassifier
from build_input import build_model_input

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

MODEL_PATH = os.getenv("MODEL_PATH", "model.pt")
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

# 10개 설문 응답을 1~5 숫자로 받는다고 가정 (네 survey_questions.py와 일치시키면 됨)
class SurveyResponse(BaseModel):
    age: int
    gender: int
    sleep_time: int
    sleep_quality: int
    fatigue: int
    diet: int
    stress: int
    digestion: int
    alcohol: int
    sunlight: int

app = FastAPI(title="Nutrient Recommend API", version="1.0")

tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model = NutrientClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

@app.get("/health")
def health():
    ok = os.path.exists(MODEL_PATH)
    return {"status": "ok" if ok else "model_missing", "device": str(DEVICE)}

@app.post("/predict")
def predict(payload: SurveyResponse):
    # build_model_input이 dict를 받아 문장을 만든다고 가정
    survey_dict = payload.model_dump()
    text = build_model_input(survey_dict)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        probs = F.softmax(logits, dim=1).squeeze()

    topk = torch.topk(probs, k=3)
    results = []
    for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
        results.append({"label_id": idx, "label": LABEL_MAP[idx], "confidence": float(score)})

    return {
        "input_text": text,
        "top3": results,
        "disclaimer": "본 결과는 정보 제공 목적이며 의료 진단/치료가 아닙니다. 기저질환/임신/약 복용 중이면 전문가와 상담하세요."
    }

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")