import os
import sys

from datasets import load_dataset
from transformers import TrainingArguments

# A1_skeleton.py lives in ../a1, not in this directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "a1"))

from A1_skeleton import (
    A1Tokenizer,
    A1Trainer,
)

from A2_skeleton import (
    A2ModelConfig,
    A2Transformer,
)


# =========================================================
# Files
# =========================================================

A1_DIR = os.path.join(os.path.dirname(__file__), "..", "a1")

TRAIN_FILE = os.path.join(A1_DIR, "train.txt")
VAL_FILE = os.path.join(A1_DIR, "val.txt")

TOKENIZER_FILE = os.path.join(A1_DIR, "trainer_output_50k", "tokenizer.pkl")

OUTPUT_DIR = "trainer_output_a2"


# =========================================================
# Load tokenizer
# =========================================================

tokenizer = A1Tokenizer.from_file(
    TOKENIZER_FILE
)

print("Vocabulary size:", len(tokenizer))


# =========================================================
# Dataset
# =========================================================

dataset = load_dataset(
    "text",
    data_files={
        "train": TRAIN_FILE,
        "val": VAL_FILE,
    }
)

dataset = dataset.filter(
    lambda x: x["text"].strip() != ""
)

print("Training examples:", len(dataset["train"]))
print("Validation examples:", len(dataset["val"]))


# =========================================================
# Model configuration
# =========================================================

config = A2ModelConfig(
    vocab_size=len(tokenizer),

    hidden_size=256,
    intermediate_size=1024,

    num_attention_heads=8,
    num_hidden_layers=2,

    rope_theta=10000.0,
    hidden_act="silu",

    max_position_embeddings=tokenizer.model_max_length,

    rms_norm_eps=1e-6,
)


model = A2Transformer(config)

print(model)

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Number of parameters: "
    f"{num_parameters:,}"
)


# =========================================================
# Training arguments
# =========================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    learning_rate=3e-4,
    num_train_epochs=3,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    optim="adamw_torch",
    eval_strategy="epoch",

    use_cpu=False,

    report_to=[],
)


# =========================================================
# Trainer
# =========================================================

trainer = A1Trainer(
    model=model,
    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],

    tokenizer=tokenizer,
)


# =========================================================
# Train
# =========================================================

trainer.train()


print(
    f"\nTraining complete. "
    f"Model saved to {OUTPUT_DIR}"
)