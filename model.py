import json
import re
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from huggingface_hub import login, HfApi, create_repo
from typing import Dict, Any, Optional
from collections import OrderedDict

# Hugging Face 설정
HF_TOKEN = os.getenv("HF_TOKEN")  # 환경 변수로 Hugging Face 토큰 설정
REPO_ID = "kimseongsan/finetuned-et5-politeness"  # Hugging Face 저장소 이름
MODEL_CARD_CONTENT = """
---
language: ko
license: apache-2.0
tags:
  - text2text-generation
  - korean
  - politeness
  - typo-correction
---

# Finetuned ET5 for Politeness and Typo Correction

This model is a fine-tuned version of `j5ng/et5-typos-corrector` for politeness enhancement and typo correction in Korean text. It transforms informal or typo-laden sentences into polite, grammatically correct ones.

## Dataset
- **Source**: Custom dataset (`last_dataset_v2.jsonl`)
- **Size**: ~300 examples
- **Task**: Converts informal/erroneous Korean sentences to polite and correct ones.

## Usage
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForSeq2SeqLM.from_pretrained("{repo_id}")

input_text = "공손화: 왜 이거 또 틀렸어요?좀"
inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)
outputs = model.generate(**inputs, max_length=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: "왜 이것을 또 틀리셨나요? 조금 더 주의해 주시면 좋겠습니다."
```

## Training
- **Base Model**: `j5ng/et5-typos-corrector`
- **Training Args**:
  - Learning Rate: 2e-5
  - Epochs: 5
  - Batch Size: 8
  - Optimizer: AdamW
- **Hardware**: GPU (e.g., NVIDIA T4)

## Limitations
- Small dataset size may lead to overfitting.
- Limited to educational context (e.g., "쌤", "숙제"). Generalization to other domains may require additional data.

## License
Apache 2.0
""".format(repo_id=REPO_ID)

# 전처리 설정
PREFIX = "공손화: "
MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 64

def clean_text(text: str) -> str:
    """텍스트를 정제하여 공백과 특수 문자를 정리합니다."""
    try:
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'([ㅠㅎ!?])\1+', r'\1', text)
        text = re.sub(r'요+', '요', text)
        return text
    except Exception as e:
        print(f"clean_text 처리 중 오류: {str(e)} | 입력 텍스트: {text}")
        return text

def preprocess_data(input_file: str) -> Dataset:
    """JSONL 파일을 로드하고 중복 제거 후 Hugging Face Dataset으로 변환."""
    data = []
    unique_inputs = set()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    cleaned_input = clean_text(entry.get('input', ''))
                    if cleaned_input not in unique_inputs:
                        unique_inputs.add(cleaned_input)
                        data.append({
                            'input': PREFIX + cleaned_input,
                            'output': clean_text(entry.get('output', ''))
                        })
                except json.JSONDecodeError:
                    print(f"유효하지 않은 JSON 라인 건너뛰기: {line.strip()}")
                    continue
    except Exception as e:
        print(f"입력 파일 읽기 중 오류 발생: {str(e)}")
        return Dataset.from_dict({})
    
    print(f"중복 제거 후 데이터 수: {len(data)}")
    return Dataset.from_list(data)

def tokenize_function(examples: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """데이터를 토크나이저로 처리."""
    try:
        inputs = tokenizer(
            examples['input'],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors=None,
            add_special_tokens=True
        )
        labels = tokenizer(
            examples['output'],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors=None,
            add_special_tokens=True
        )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']
        }
    except Exception as e:
        print(f"토크나이징 중 오류: {str(e)} | 입력: {examples}")
        return {
            'input_ids': [tokenizer.pad_token_id] * MAX_INPUT_LENGTH,
            'attention_mask': [0] * MAX_INPUT_LENGTH,
            'labels': [tokenizer.pad_token_id] * MAX_TARGET_LENGTH
        }

def upload_to_hub(model, tokenizer, repo_id: str, token: Optional[str] = None):
    """모델과 토크나이저를 Hugging Face Hub에 업로드."""
    try:
        # Hugging Face 로그인
        if token:
            login(token=token)
        else:
            print("HF_TOKEN 환경 변수가 필요합니다.")
            return
        
        # 저장소 생성 (이미 존재하면 무시)
        create_repo(repo_id=repo_id, exist_ok=True, private=False)  # private=True로 비공개 설정 가능
        
        # 모델과 토크나이저 업로드
        model.push_to_hub(repo_id, commit_message="Upload fine-tuned model")
        tokenizer.push_to_hub(repo_id, commit_message="Upload tokenizer")
        
        # 모델 카드 작성
        with open(os.path.join("README.md"), "w", encoding="utf-8") as f:
            f.write(MODEL_CARD_CONTENT)
        api = HfApi()
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Add model card"
        )
        
        print(f"모델과 토크나이저가 {repo_id}에 성공적으로 업로드되었습니다.")
    except Exception as e:
        print(f"Hugging Face Hub 업로드 중 오류: {str(e)}")

def main():
    # 입력 파일 및 출력 디렉토리 경로
    input_file = '/home/ubuntu/Seungsan/capstone/last_LLM_model/last_dataset_v2.jsonl'
    output_dir = '/home/ubuntu/Seungsan/capstone/output/finetuned_et5'
    
    # Hugging Face 토큰 확인
    if not HF_TOKEN:
        print("환경 변수 HF_TOKEN을 설정해주세요. 예: export HF_TOKEN='your_token_here'")
        return
    
    # 토크나이저와 모델 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained('j5ng/et5-typos-corrector')
        model = AutoModelForSeq2SeqLM.from_pretrained('j5ng/et5-typos-corrector')
    except Exception as e:
        print(f"모델/토크나이저 로드 중 오류: {str(e)}")
        return
    
    # 데이터셋 로드 및 전처리
    dataset = preprocess_data(input_file)
    if len(dataset) == 0:
        print("데이터셋이 비어 있습니다. 프로그램 종료.")
        return
    
    # 데이터셋 분할 (80% 훈련, 20% 검증)
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    # 토크나이징
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['input', 'output'],
        desc="Tokenizing training dataset"
    )
    tokenized_eval = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['input', 'output'],
        desc="Tokenizing evaluation dataset"
    )
    
    # 학습 인자 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True if torch.cuda.is_available() else False,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )
    
    # 트레이너 설정
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
    )
    
    # 학습 시작
    try:
        trainer.train()
        print(f"모델이 {output_dir}에 저장되었습니다.")
    except Exception as e:
        print(f"학습 중 오류 발생: {str(e)}")
        return
    
    # 모델 및 토크나이저 저장
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"최종 모델과 토크나이저가 {output_dir}에 저장되었습니다.")
    except Exception as e:
        print(f"모델 저장 중 오류: {str(e)}")
        return
    
    # Hugging Face Hub에 업로드
    upload_to_hub(model, tokenizer, REPO_ID, HF_TOKEN)

if __name__ == "__main__":
    main()
