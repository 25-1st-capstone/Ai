import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load  # evaluate 라이브러리 사용
from typing import List, Tuple

# 모델 및 토크나이저 설정
REPO_ID = "kimseongsan/finetuned-et5-politeness"
PREFIX = "공손화: "
MAX_INPUT_LENGTH = 64
MAX_OUTPUT_LENGTH = 64

def load_model_and_tokenizer(repo_id: str):
    """모델과 토크나이저를 Hugging Face Hub에서 로드."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
        if torch.cuda.is_available():
            model = model.cuda()
        print(f"모델과 토크나이저가 {repo_id}에서 성공적으로 로드되었습니다.")
        return tokenizer, model
    except Exception as e:
        print(f"모델/토크나이저 로드 중 오류: {str(e)}")
        return None, None

def generate_polite_text(input_text: str, tokenizer, model, max_length: int = MAX_OUTPUT_LENGTH) -> str:
    """입력 문장을 공손화 및 맞춤법 교정된 문장으로 생성."""
    try:
        input_text = PREFIX + input_text
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length"
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,  # 부자연스러운 출력 방지
            no_repeat_ngram_size=2,  # 반복 단어 방지
            early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"생성 중 오류: {str(e)} | 입력: {input_text}")
        return ""

def evaluate_bleu(predictions: List[str], references: List[str]) -> float:
    """BLEU 점수를 계산하여 출력 품질 평가."""
    try:
        bleu = load("bleu")  # evaluate 라이브러리에서 BLEU 로드
        predictions = [[pred.split()] for pred in predictions]
        references = [[[ref.split()]] for ref in references]
        result = bleu.compute(predictions=predictions, references=references)
        return result["bleu"]
    except Exception as e:
        print(f"BLEU 계산 중 오류: {str(e)}")
        return 0.0

def main():
    # 모델과 토크나이저 로드
    tokenizer, model = load_model_and_tokenizer(REPO_ID)
    if tokenizer is None or model is None:
        return
    
    # 테스트 케이스 (학습 데이터 유사 + 새로운 맥락)
    test_cases = [
        ("왜 이거 또 틀렸어요?좀", "왜 이것을 또 틀리셨나요? 조금 더 주의해 주시면 좋겠습니다."),  # 학습 데이터 유사
        ("아 몰라요?ㅠㅠ", "잘 모르겠는데요?"),  # 학습 데이터 유사
        ("이거 진짜 짜증나", "이 상황이 정말 불편하시군요."),  # 새로운 맥락
        ("쌤 이거 왜 안돼요?", "선생님, 이 문제가 왜 작동하지 않나요?"),  # 학습 데이터 유사
        ("빨리 해줘", "빠르게 처리해 주시면 감사하겠습니다.")  # 새로운 맥락
    ]
    
    # 결과 저장
    predictions = []
    references = []
    
    # 테스트 실행
    print("\n=== 모델 테스트 결과 ===")
    for input_text, expected_output in test_cases:
        output = generate_polite_text(input_text, tokenizer, model)
        print(f"입력: {input_text}")
        print(f"출력: {output}")
        print(f"예상 출력: {expected_output}")
        print("-" * 50)
        predictions.append(output)
        references.append(expected_output)
    
    # BLEU 점수 계산
    bleu_score = evaluate_bleu(predictions, references)
    print(f"BLEU 점수: {bleu_score:.4f}")
    
    # 추가 테스트: 사용자 입력
    print("\n=== 사용자 입력 테스트 ===")
    while True:
        user_input = input("테스트할 문장을 입력하세요 (종료: 'quit'): ")
        if user_input.lower() == 'quit':
            break
        output = generate_polite_text(user_input, tokenizer, model)
        print(f"입력: {user_input}")
        print(f"출력: {output}")
        print("-" * 50)

if __name__ == "__main__":
    main()