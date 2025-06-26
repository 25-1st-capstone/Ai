
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import evaluate
import numpy as np
import json
from huggingface_hub import notebook_login

# ✅ 0. 로그인 (한 번만 실행하면 토큰 저장됨)
notebook_login()

# ✅ 1. 사전학습된 모델 불러오기
model_name = "KETI-AIR/ke-t5-base-ko"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ✅ 2. 데이터 로딩 (형식: [{"input": "...", "output": "..."}])
with open("politeness_dataset_1000.json", encoding="utf-8") as f:
    raw_data = json.load(f)
dataset = Dataset.from_list(raw_data)

# ✅ 3. 전처리
prefix = "공손화: "
max_input_length = 64
max_target_length = 64

def preprocess(example):
    input_text = prefix + example["input"]
    inputs = tokenizer(input_text, max_length=max_input_length, truncation=True)
    labels = tokenizer(example["output"], max_length=max_target_length, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, remove_columns=["input", "output"])

# ✅ 4. 학습 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned-kot5",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=True,
    hub_model_id="seongsan/polite-corrector-ko",  # 업로드될 모델 이름
    hub_strategy="every_save"
)

# ✅ 5. 평가 함수 (ROUGE)
rouge = evaluate.load("rouge")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}

# ✅ 6. 트레이너 구성
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(range(100)),
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    compute_metrics=compute_metrics
)

# ✅ 7. 학습 시작
trainer.train()

# ✅ 8. 최종 모델 저장 및 업로드
trainer.push_to_hub()
