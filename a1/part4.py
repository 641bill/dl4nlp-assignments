import os
import sys
import torch
import nltk

from datasets import load_dataset
from torch.utils.data import Subset
from transformers import TrainingArguments

from A1_skeleton import (
    build_tokenizer,
    A1Tokenizer,
    A1RNNModelConfig,
    A1RNNModel,
    A1Trainer,
)


# ============================================================
# Experiment configuration
# ============================================================

TRAIN_FILE = "train.txt"
VAL_FILE = "val.txt"

# Change these to run a different experiment without overwriting others.
MAX_VOC_SIZE = 50_000
OUTPUT_DIR = "trainer_output_50k"
TOKENIZER_FILE = os.path.join(OUTPUT_DIR, "tokenizer.pkl")

MODEL_MAX_LENGTH = 128
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256

LEARNING_RATE = 1e-3
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

# Set to True to run a quick training-loop sanity check before full training.
RUN_SANITY_CHECK = False
SANITY_TRAIN_SIZE = 128
SANITY_VAL_SIZE = 32
SANITY_EPOCHS = 1


# ============================================================
# Helpers
# ============================================================

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_sanity_check(tokenizer, train_dataset, val_dataset):
    """Verify forward, loss, backward, and optimizer step on a tiny subset."""
    print("\n" + "-" * 60)
    print("SANITY CHECK (tiny subset, 1 epoch)")
    print("-" * 60)

    device = select_device()
    print("Device:", device)

    config = A1RNNModelConfig(
        vocab_size=len(tokenizer),
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
    )
    model = A1RNNModel(config).to(device)
    assert model.config.vocab_size == len(tokenizer)

    sanity_train = Subset(train_dataset, range(min(SANITY_TRAIN_SIZE, len(train_dataset))))
    sanity_val = Subset(val_dataset, range(min(SANITY_VAL_SIZE, len(val_dataset))))

    sanity_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "_sanity_tmp"),
        optim="adamw_torch",
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        num_train_epochs=SANITY_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        use_cpu=(device.type == "cpu"),
        disable_tqdm=True,
        report_to="none",
    )

    trainer = A1Trainer(
        model=model,
        args=sanity_args,
        train_dataset=sanity_train,
        eval_dataset=sanity_val,
        tokenizer=tokenizer,
    )
    trainer.train()

    # Forward-pass check on a single prompt.
    model.eval()
    encoding = tokenizer(
        ["She lives in San"],
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(encoding["input_ids"].to(device))
    assert torch.isfinite(output.logits).all()

    print("Sanity check passed.\n")


def save_experiment_info(tokenizer, num_parameters):
    info_path = os.path.join(OUTPUT_DIR, "experiment_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: up-to-{MAX_VOC_SIZE} vocabulary RNN language model\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"  max_voc_size (limit)     = {MAX_VOC_SIZE}\n")
        f.write(f"  actual_vocabulary_size   = {len(tokenizer)}\n")
        f.write(f"  embedding_size           = {EMBEDDING_SIZE}\n")
        f.write(f"  hidden_size              = {HIDDEN_SIZE}\n")
        f.write(f"  model_max_length         = {MODEL_MAX_LENGTH}\n")
        f.write(f"  learning_rate            = {LEARNING_RATE}\n")
        f.write(f"  num_epochs               = {NUM_EPOCHS}\n")
        f.write(f"  train_batch_size         = {TRAIN_BATCH_SIZE}\n")
        f.write(f"  eval_batch_size          = {EVAL_BATCH_SIZE}\n")
        f.write(f"  total_parameters         = {num_parameters:,}\n")
    print(f"Experiment info saved to: {info_path}")


def main():
    print("=" * 60)
    print(f"A1 FINAL TRAINING  (max_voc_size={MAX_VOC_SIZE:,})")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Cannot find {TRAIN_FILE}")
    if not os.path.exists(VAL_FILE):
        raise FileNotFoundError(f"Cannot find {VAL_FILE}")

    if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        print(
            f"\nWARNING: {OUTPUT_DIR}/ already exists and is non-empty.\n"
            "Training will overwrite model files in that directory.\n"
            "The preserved 10k experiment in trainer_output_10k/ will NOT be touched.\n"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    # --------------------------------------------------------
    # 1. Build tokenizer
    # --------------------------------------------------------
    print("\n[1/6] Building tokenizer...")
    tokenizer = build_tokenizer(
        train_file=TRAIN_FILE,
        max_voc_size=MAX_VOC_SIZE,
        model_max_length=MODEL_MAX_LENGTH,
    )
    print("Vocabulary size:", len(tokenizer))
    print("Model max length:", tokenizer.model_max_length)
    print("PAD token ID:", tokenizer.pad_token_id)
    print("UNK token ID:", tokenizer.unk_token_id)
    print("BOS token ID:", tokenizer.bos_token_id)
    print("EOS token ID:", tokenizer.eos_token_id)

    tokenizer.save(TOKENIZER_FILE)
    print(f"Tokenizer saved to: {TOKENIZER_FILE}")

    # --------------------------------------------------------
    # 2. Load full datasets
    # --------------------------------------------------------
    print("\n[2/6] Loading full datasets...")
    dataset = load_dataset(
        "text",
        data_files={"train": TRAIN_FILE, "val": VAL_FILE},
    )
    dataset = dataset.filter(lambda x: x["text"].strip() != "")

    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))

    # --------------------------------------------------------
    # 3. Create model
    # --------------------------------------------------------
    print("\n[3/6] Creating model...")
    config = A1RNNModelConfig(
        vocab_size=len(tokenizer),
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
    )
    model = A1RNNModel(config)

    num_parameters = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Vocabulary size:", config.vocab_size)
    print("Embedding size:", config.embedding_size)
    print("Hidden size:", config.hidden_size)
    print(f"Total parameters: {num_parameters:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # --------------------------------------------------------
    # 4. Optional sanity check
    # --------------------------------------------------------
    if RUN_SANITY_CHECK:
        run_sanity_check(tokenizer, train_dataset, val_dataset)

    # --------------------------------------------------------
    # 5. Configure training
    # --------------------------------------------------------
    print("\n[4/6] Configuring training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        optim="adamw_torch",
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        use_cpu=False,
        disable_tqdm=True,
        report_to="none",
    )
    print("Learning rate:", LEARNING_RATE)
    print("Epochs:", NUM_EPOCHS)
    print("Training batch size:", TRAIN_BATCH_SIZE)
    print("Evaluation batch size:", EVAL_BATCH_SIZE)

    # --------------------------------------------------------
    # 6. Full training
    # --------------------------------------------------------
    print("\n[5/6] Starting full training on FULL datasets...")
    print("This is the real training run.\n")

    trainer = A1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    save_experiment_info(tokenizer, num_parameters)

    # --------------------------------------------------------
    # 7. Verify saved artifacts
    # --------------------------------------------------------
    print("\n[6/6] Verifying saved model...")
    if not os.path.isdir(OUTPUT_DIR):
        raise RuntimeError(f"Output directory {OUTPUT_DIR} was not created.")
    if not os.path.exists(TOKENIZER_FILE):
        raise RuntimeError(f"Tokenizer was not saved to {TOKENIZER_FILE}.")

    loaded_model = A1RNNModel.from_pretrained(OUTPUT_DIR)
    loaded_tokenizer = A1Tokenizer.from_file(TOKENIZER_FILE)
    assert loaded_model.config.vocab_size == len(loaded_tokenizer)

    encoding = loaded_tokenizer(
        ["She lives in San"],
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    loaded_model.eval()
    with torch.no_grad():
        output = loaded_model(encoding["input_ids"])
    assert output.logits.shape[0] == 1
    assert output.logits.shape[1] == encoding["input_ids"].shape[1]
    assert output.logits.shape[2] == len(loaded_tokenizer)

    print("Saved model reload: OK")
    print("Saved tokenizer reload: OK")
    print("Reloaded model forward pass: OK")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"""
Model and tokenizer saved under:

    {OUTPUT_DIR}/
    {TOKENIZER_FILE}

Preserved 10k experiment (unchanged):

    trainer_output_10k/
""")


if __name__ == "__main__":
    main()
