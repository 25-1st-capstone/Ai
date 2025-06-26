from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = FastAPI()

# ✅ 사용자 모델 설정
model_name = "kimseongsan/polite_corrector_ko_last"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ✅ GPU 지원
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ 규칙 기반 판단 함수 (공손하지 않거나 비문 판단)
def is_problematic(text: str) -> bool:
    informal_words = ["쌤", "못감", "몰라", "뭐야", "야", "가자", "왜그래", "꺼져", "ㅋㅋ", "ㅎㅇ"]
    informal_endings = ["해", "했어", "냐", "거든", "간다", "말야", "잖아", "지?"]
    if any(word in text for word in informal_words):
        return True
    if text.strip().endswith(tuple(informal_endings)):
        return True
    return False

# ✅ 입력 스키마
class Question(BaseModel):
    text: str

# ✅ API 엔드포인트
@app.post("/ask")
def ask(question: Question):
    prompt = f"문장을 공손하고 자연스럽게 맞춤법까지 고쳐줘: {question.text}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output_ids = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
        no_repeat_ngram_size=4,
        repetition_penalty=1.8,
        temperature=0.7,
        top_p=0.9,
        early_stopping=True,
        decoder_start_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id
    )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {
        "input": question.text,
        "corrected": is_problematic(question.text),
        "answer": answer
    }
