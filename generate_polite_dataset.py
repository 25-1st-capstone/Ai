
import json
import random
import argparse

templates = [
    ("쌤 저 {subject} 숙제 안했어요", "선생님, 아직 {subject} 숙제를 제출하지 못했습니다."),
    ("쌤 나 {reason} 때문에 못감", "선생님, {reason} 때문에 출석하지 못했습니다."),
    ("{activity} 언제까지 내야됨?", "{activity}는 언제까지 제출해야 하나요?"),
    ("{time} 수업 언제 시작함?", "{time} 수업은 언제 시작하나요?"),
    ("쌤 오늘 수업 있음?", "선생님, 오늘 수업이 있는지 여쭈어봐도 될까요?"),
    ("시험 범위 어디까지임?", "시험 범위가 어디까지인지 알려주실 수 있을까요?"),
    ("쌤 내일 조퇴 가능?", "선생님, 내일 조퇴해도 괜찮을까요?"),
    ("이거 정답 뭐에요?", "선생님, 이 문제의 정답이 무엇인지 알려주실 수 있을까요?"),
    ("{event} 언제에요?", "{event}는 언제 진행되나요?"),
    ("쌤 {place} 가도 돼요?", "선생님, {place}에 가도 괜찮을까요?")
]

subjects = ["영어", "수학", "과학", "국어", "사회"]
reasons = ["감기", "몸살", "병원 예약", "가족 행사"]
activities = ["과제", "보고서", "자기소개서"]
times = ["1교시", "2교시", "오전 수업"]
events = ["운동회", "소풍", "시험", "현장학습"]
places = ["보건실", "상담실", "도서관", "교무실"]

def generate_dataset(n=1000):
    data = []
    seen = set()
    while len(data) < n:
        temp = random.choice(templates)
        input_text = temp[0].format(
            subject=random.choice(subjects),
            reason=random.choice(reasons),
            activity=random.choice(activities),
            time=random.choice(times),
            event=random.choice(events),
            place=random.choice(places)
        )
        output_text = temp[1].format(
            subject=random.choice(subjects),
            reason=random.choice(reasons),
            activity=random.choice(activities),
            time=random.choice(times),
            event=random.choice(events),
            place=random.choice(places)
        )
        key = (input_text, output_text)
        if key not in seen:
            seen.add(key)
            data.append({"input": input_text, "output": output_text})
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="dataset.json", help="Output filename")
    args = parser.parse_args()

    dataset = generate_dataset(args.count)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {args.count} samples and saved to '{args.output}'")
