"""
Script to evaluate the accuracy of the student student with ROUGE.

It compares the ROUGE scoreds of the teacher and the student and
generates accuracy values.

"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge import Rouge
from datasets import load_dataset
from tqdm import tqdm

device = torch.device("cpu")


class RougeScores:
    """Keeps track of Rouge scores for a given model."""

    def __init__(self, model_id, tokenizer_id):
        self._scores = {}
        self._num = 0
        self._model = T5ForConditionalGeneration.from_pretrained(model_id)
        self._model.to(device)
        self._tokenizer = T5Tokenizer.from_pretrained(tokenizer_id)
        self._scorer = Rouge()

    def _summarize(self, text):
        """Given a model and a text, returns a summary."""
        self._model.eval()

        input_ids = self._tokenizer.encode(
            "summarize: " + text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        ).to(device)

        generated_ids = self._model.generate(
            input_ids, max_length=130, min_length=30, do_sample=False
        )[0]

        return self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            remove_invalid_values=True,
        )

    def evaluate(self, content, reference_summary):
        summary = self._summarize(content).strip()
        if summary == "":
            print("Summary is empty!")
            return
        scores = self._scorer.get_scores(summary, reference_summary)
        self._add(scores[0])

    def _add(self, source):
        for metric, scores in source.items():
            if metric not in self._scores:
                self._scores[metric] = scores
            else:
                for name, value in scores.items():
                    if name not in self._scores[metric]:
                        self._scores[metric][name] = value
                    else:
                        self._scores[metric][name] += value
        self._num += 1

    def avg(self):
        avg_scores = {}
        for metric, scores in self._scores.items():
            avg_scores[metric] = {}
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

    def print_accuracy(self, teacher):
        acc = self.accuracy(teacher)

        def replace(name):
            if name == "f":
                return "F1"
            if name == "p":
                return "Precision"
            return "Recall"

        for metric, scores in acc.items():
            print(f"{metric} Accuracy:")
            for name, value in scores.items():
                print(f"- {replace(name)} Accuracy: {value:.2f}%")
            print()


teacher = RougeScores(
    "JulesBelveze/t5-small-headline-generator",
    "JulesBelveze/t5-small-headline-generator",
)

student = RougeScores(
    "./t5-small-headline-generator-sft-3", "JulesBelveze/t5-small-headline-generator"
)


dataset = load_dataset("JulesBelveze/tldr_news")

for line in tqdm(dataset["test"]):
    content = line["content"].strip()
    headline = line["headline"].strip()

    if not content or not headline:
        continue

    teacher.evaluate(content, headline)
    student.evaluate(content, headline)


student.print_accuracy(teacher)
