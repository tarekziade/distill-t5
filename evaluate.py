"""
Script to evaluate the accuracy of the student student with ROUGE.

It compares the ROUGE scoreds of the teacher and the student and
generates accuracy values.

"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge import Rouge
from transformers import pipeline
from pprint import pprint
from datasets import load_dataset
from tqdm import tqdm


device = torch.device("cpu")


student = T5ForConditionalGeneration.from_pretrained(
    "./t5-small-headline-generator-sft-3"
)
student.to(device)

teacher = T5ForConditionalGeneration.from_pretrained(
    "JulesBelveze/t5-small-headline-generator"
)
teacher.to(device)

tokenizer = T5Tokenizer.from_pretrained("JulesBelveze/t5-small-headline-generator")


def summarize(student, text):
    student.eval()

    input_ids = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    ).to(device)

    generated_ids = student.generate(
        input_ids, max_length=130, min_length=30, do_sample=False
    )[0]
    summary = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        remove_invalid_values=True,
    )

    return summary


class RougeScores:
    def __init__(self):
        self._scores = {}
        self._num = 0

    def evaluate(self, summary, reference_summary):
        score = self._evaluate(summary, reference_summary)
        self._add(score)

    def _evaluate(summary, reference_summary):
        rouge = Rouge()
        scores = rouge.get_scores(summary, reference_summary)
        return scores[0]

    def _add(self, source):
        for metric, scores in source.items():
            if metric not in self._scores:
                self._scores[metric] = scores
            else:
                for name, value in scores.items():
                    if name not in target[metric]:
                        self._scores[metric][name] = value
                    else:
                        self._scores[metric][name] += value
        self._num += 1

    def avg(self):
        avg_scores = {}
        for metric, scores in self._scores.items():
            for name, value in scores.items():
                avg_scores[metric][name] = value / self._num
        return avg_scores

    def accuracy(self, teacher):
        teacher_scores = teacher.avg()
        student_scores = self.avg()
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


teacher_scores = RougeScores()
student_scores = RougeScores()


dataset = load_dataset("JulesBelveze/tldr_news")
for line in tqdm(dataset["test"]):
    content = line["content"].strip()
    headline = line["headline"].strip()

    if not content or not headline:
        continue
    try:
        teacher_scores.evaluate(summarize(teacher, content), headline)
        student_scores.evaluate(summarize(student, content), headline)
    except Exception:
        pass


accuracy = student.accuracy(teacher)


def replace(name):
    if name == "f":
        return "F1"
    if name == "p":
        return "Precision"
    return "Recall"


for metric, scores in accuracy.items():
    print(f"{metric} Accuracy:")
    for name, value in scores.items():
        print(f"- {replace(name)} Accuracy: {value:.2f}%")
    print()
