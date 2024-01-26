"""
Shrink and Fine-tune a T5 model for summarization.

1. remove 50% of the decoder layers
2. fine-tune the shrinked model with various datasets
"""
import os
import functools

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from datasets import (
    load_dataset,
    load_metric,
    concatenate_datasets,
)
import numpy as np
import nltk
import re

nltk.download("punkt")


TEST_DATA = """\
The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
"""


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_size(model):
    size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
    return "%.2fM" % size


def load_and_shrink_t5_model(model_name, tokenizer, layers=None):
    shrunk_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    current_decoder_layers = shrunk_model.config.num_decoder_layers
    if layers is None:
        new_size = int(current_decoder_layers / 2)
    else:
        new_size = layers

    print(
        f"This model has {current_decoder_layers} layers, and {get_size(shrunk_model)} parameters shrinking in half"
    )

    test_model("Before Shrink, before fine-tuning: ", shrunk_model, tokenizer)

    layers_to_remove = [i for i in range(1, new_size * 2, 2)]

    for i in reversed(layers_to_remove):
        del shrunk_model.decoder.block[i]

    shrunk_model.config.num_decoder_layers = new_size
    print(
        f"This model has now {new_size} layers and {get_size(shrunk_model)} parameters"
    )

    return shrunk_model


def compute_metrics(eval_pred, tokenizer):
    rouge = load_metric("rouge")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    return rouge.compute(predictions=decoded_preds, references=decoded_labels)


def load_and_tokenize_dataset(tokenizer):
    def sentence_aware_truncation(text, max_length):
        sentences = nltk.tokenize.sent_tokenize(text)
        truncated_text = ""
        for sentence in sentences:
            tokens = tokenizer.encode(
                truncated_text + sentence, add_special_tokens=False
            )
            if len(tokens) > max_length:
                break
            truncated_text += sentence + " "

        return truncated_text.strip()

    def tokenize_function(example, source_field="chapter", target_field="summary_text"):
        inputs = tokenizer(
            [
                sentence_aware_truncation("summarize: " + clean_text(text), 512)
                for text in example[source_field]
            ],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

        targets = tokenizer(
            [
                sentence_aware_truncation(clean_text(summary), 200)
                for summary in example[target_field]
            ],
            padding="max_length",
            truncation=True,
            max_length=200,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": targets.input_ids,
        }

    def prepare_dataset(dataset, source_field, summary_field):
        dataset = dataset.filter(
            lambda x: x[source_field] is not None and x[summary_field] is not None
        )

        dataset = dataset.map(
            functools.partial(
                tokenize_function, source_field=source_field, target_field=summary_field
            ),
            batched=True,
        )

        dataset = dataset.select_columns(
            [
                "input_ids",
                "attention_mask",
                "labels",
            ]
        )
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return dataset

    # print("Loading Booksum")
    # train_dataset = load_dataset("kmfoda/booksum")
    # train_dataset = prepare_dataset(train_dataset, "chapter", "summary_text")

    # adding more data to the test split from Wikipedia, xsum and tldr_news
    # for dataset, split, source_field, target_field in (
    #    ("tarekziade/wikipedia-topics", "train[:20%]", "text", "summary"),
    #    ("EdinburghNLP/xsum", "train[:5%]", "document", "summary"),
    #    ("JulesBelveze/tldr_news", "train[:50%]", "content", "headline"),
    # ):
    #    print(f"Loading {dataset}")
    #    extra_dataset = load_dataset(dataset, split=split)
    #    extra_dataset = prepare_dataset(extra_dataset, source_field, target_field)
    #    train_dataset["train"] = concatenate_datasets(
    #        [train_dataset["train"], extra_dataset]
    #    )
    train_dataset = load_dataset("JulesBelveze/tldr_news")
    train_dataset = prepare_dataset(train_dataset, "content", "headline")
    return train_dataset.shuffle()


class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"current state.best_metric: {state.best_metric}")
        print(
            f"current early_stopping_patience_counter: {self.early_stopping_patience_counter}"
        )

        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)
        print(
            f"metric value: {metric_value} for metric: {metric_to_check} and state.best_metric: {state.best_metric}"
        )

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None:
            self.early_stopping_patience_counter = 0
            return

        val = operator(metric_value, state.best_metric)
        diff = abs(metric_value - state.best_metric)
        print(f"less(metric_value, state.best_metric) == {val}")
        print(f"abs(metric_value - state.best_metric) == {diff}")

        if val and diff > self.early_stopping_threshold:
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1


def fine_tune_model(model, tokenizer):
    tokenized_datasets = load_and_tokenize_dataset(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        run_name="shrink-and-finetune",
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # eval_steps=10,
        # save_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        # callbacks=[
        #    CustomEarlyStoppingCallback(
        #        early_stopping_patience=5, early_stopping_threshold=0.1
        #    )
        # ]
        # compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    trainer.train()
    trainer.eval_dataset = tokenized_datasets["test"]
    metrics = trainer.evaluate()
    print(metrics)


def test_model(prefix, model, tokenizer):
    model.eval()

    input_ids = tokenizer.encode(
        "summarize: " + clean_text(TEST_DATA),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        add_special_tokens=False,
    ).to(device)

    generated_ids = model.generate(
        input_ids, max_length=120, no_repeat_ngram_size=0, num_beams=4, top_k=50
    )[0]
    print(
        "*********************** "
        + prefix
        + ": \n"
        + tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            remove_invalid_values=True,
        )
        + "***********************"
        + "\n\n"
    )


if __name__ == "__main__":
    # model_name = "cnicu/t5-small-booksum"
    # model_name = "Alred/t5-small-finetuned-summarization-cnn"
    # model_name = "Falconsai/text_summarization"
    # model_name = "flax-community/t5-base-cnn-dm"
    # tokenizer_name = "t5-base"
    os.environ["WANDB_PROJECT"] = "shrink-and-fine-tune"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    model_name = tokenizer_name = "JulesBelveze/t5-small-headline-generator"
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    shrunk_model = load_and_shrink_t5_model(model_name, tokenizer)

    test_model("After Shrink, before fine-tuning: ", shrunk_model, tokenizer)
    try:
        fine_tune_model(shrunk_model, tokenizer)
    finally:
        test_model("After Shrink and fine-tuning ", shrunk_model, tokenizer)

        target_dir = (
            f"{model_name.split('/')[-1]}-sft-{shrunk_model.config.num_decoder_layers}"
        )

        shrunk_model.save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)

        # TODO: save the onnx quantized version inside target_dir/onnx for transformers.js
