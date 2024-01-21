import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import log_softmax, softmax
from torch import nn
from torch import tensor

mps_device = torch.device("mps")


TEST_DATA = """\
The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
"""
teacher_model = T5ForConditionalGeneration.from_pretrained("cnicu/t5-small-booksum")
teacher_model.to(mps_device)

teacher_model.eval()

tokenizer = T5Tokenizer.from_pretrained("cnicu/t5-small-booksum")

train_dataset = load_dataset("kmfoda/booksum", split="train")
train_dataset = train_dataset.select_columns(
    [
        "summary_length",
        "summary_text",
        "chapter",
    ]
)
train_dataset = train_dataset.filter(
    lambda x: x["chapter"] is not None and x["summary_text"] is not None
)


def test_model(prefix, model):
    model.eval()

    input_ids = tokenizer.encode(
        TEST_DATA,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    ).to(mps_device)

    generated_ids = model.generate(input_ids, max_length=120)[0]
    print(
        prefix
        + " "
        + tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            remove_invalid_values=True,
        )
    )


def tokenize_function(example):
    inputs = tokenizer(
        example["chapter"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    targets = tokenizer(
        example["summary_text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": targets.input_ids,
    }


train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.remove_columns(
    ["chapter", "summary_text", "summary_length"]
)


train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

print(train_dataset)

config = teacher_model.config
# config.num_layers = 2
# config.d_model = 128
# config.d_ff = 512
# config.d_kv = 64

student_model = T5ForConditionalGeneration(config)
student_model.to(mps_device)


# Hyperparameters
learning_rate = 0.005
batch_size = 16
num_epochs = 13
temperature = 20
alpha = 0.7


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)


def calculate_loss(student_outputs, teacher_outputs, labels):
    s_logits = student_outputs.logits
    t_logits = teacher_outputs.logits

    vocab_size = s_logits.size(-1)
    ce_logits = s_logits.view(-1, vocab_size)
    ce_labels = labels.view(-1)
    ce_loss = torch.nn.functional.cross_entropy(ce_logits, ce_labels)
    student_log_probs = log_softmax(s_logits.view(-1, vocab_size) / temperature, dim=-1)
    teacher_probs = softmax(t_logits.view(-1, vocab_size) / temperature, dim=-1)

    distill_loss = torch.nn.functional.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    )
    loss = (1 - alpha) * ce_loss + (
        alpha * temperature**2 / batch_size**2
    ) * distill_loss

    return loss


# Training Loop
for epoch in range(num_epochs):
    loss_value = 0

    test_model("Before epoch " + str(epoch), student_model)
    student_model.train()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        optimizer.zero_grad()

        batch = dict([(k, v.to(mps_device)) for k, v in batch.items()])

        # Forward pass through the teacher model
        with torch.no_grad():
            teacher_outputs = teacher_model(**batch)

        # Forward pass through the student model
        student_outputs = student_model(**batch)
        assert student_outputs.logits.size() == teacher_outputs.logits.size()
        loss = calculate_loss(student_outputs, teacher_outputs, batch["labels"])
        # Backpropagation
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss_value=loss.item())


student_model.save_pretrained("distilled-t5-small")

test_model("After distillation ", student_model)
