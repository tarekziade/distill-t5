import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge import Rouge
from transformers import pipeline
from pprint import pprint
from datasets import load_dataset
from tqdm import tqdm


device = torch.device("cpu")


model = T5ForConditionalGeneration.from_pretrained(
    "./t5-small-headline-generator-sft-3"
)
model.to(device)

original = T5ForConditionalGeneration.from_pretrained(
    "JulesBelveze/t5-small-headline-generator"
)
original.to(device)

tokenizer = T5Tokenizer.from_pretrained("JulesBelveze/t5-small-headline-generator")
dataset = load_dataset("JulesBelveze/tldr_news")


def test_model(model, text):
    model.eval()

    input_ids = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    ).to(device)

    generated_ids = model.generate(
        input_ids, max_length=130, min_length=30, do_sample=False
    )[0]
    summary = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        remove_invalid_values=True,
    )

    return summary


teacher_scores = {}
student_scores = {}


def evaluate(summary, reference_summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference_summary)
    return scores[0]


def add(source, target):
    for metric, scores in source.items():
        if metric not in target:
            target[metric] = scores
        else:
            for name, value in scores.items():
                if name not in target[metric]:
                    target[metric][name] = value
                else:
                    target[metric][name] += value


def avg(source, num):
    for metric, scores in source.items():
        for name, value in scores.items():
            source[metric][name] = value / num


num = 0
for line in tqdm(dataset["test"]):
    content = line["content"]
    headline = line["headline"]
    if not content or not headline:
        continue
    try:
        summary = test_model("original", original, content)
        or_ = evaluate(summary, headline)
        add(or_, teacher_scores)

        shrinked_summary = test_model("shrinked", model, content)
        sr = evaluate(shrinked_summary, headline)
        add(sr, student_scores)
        num += 1
    except Exception:
        pass


avg(teacher_scores, num)
avg(student_scores, num)

print(teacher_scores)
print(student_scores)


def calculate_student_accuracy(teacher_scores, student_scores):
    metrics = ["rouge-1", "rouge-2", "rouge-l"]
    components = ["f", "p", "r"]
    accuracy_scores = {}

    for metric in metrics:
        accuracy_scores[metric] = {}
        for component in components:
            teacher_score = teacher_scores[metric][component]
            student_score = student_scores[metric][component]

            # Calculate the difference and convert it to a percentage
            difference = abs(teacher_score - student_score)
            accuracy = ((1 - difference) / 1) * 100  # Convert to percentage
            accuracy_scores[metric][component] = accuracy

    return accuracy_scores


def replace(name):
    if name == "f":
        return "F1"
    if name == "p":
        return "Precision"
    return "Recall"


accuracy = calculate_student_accuracy(teacher_scores, student_scores)
for metric, scores in accuracy.items():
    print(f"{metric} Accuracy:")
    for name, value in scores.items():
        print(f"- {replace(name)} Accuracy: {value:.2f}%")
    print()
