"""Evaluate trained A2 Transformer (Task 2.1) — reload, next-word, perplexity."""

import json
import math
import os
import sys

# Use project-local HF cache (avoids permission issues in some environments).
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    os.path.join(os.path.dirname(__file__), ".datasets_cache"),
)

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

A1_DIR = os.path.join(os.path.dirname(__file__), "..", "a1")
sys.path.insert(0, A1_DIR)

from A1_skeleton import A1Tokenizer
from A2_skeleton import A2Transformer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "final_model")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.pkl")
VAL_FILE = os.path.join(A1_DIR, "val.txt")
EVAL_BATCH_SIZE = 8

PROMPTS = [
    "She lives in San",
    "The capital of Sweden is",
    "The cat sat on the",
]


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class A2DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        texts = [ex["text"] for ex in examples]
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}


def predict_next_words(model, tokenizer, device, prompt, k=10):
    encoding = tokenizer(
        [prompt],
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    with torch.no_grad():
        output = model(input_ids)
    pos = -2 if input_ids[0, -1].item() == tokenizer.eos_token_id else -1
    probs = torch.softmax(output.logits[0, pos, :], dim=-1)
    top_p, top_i = torch.topk(probs, k=k)
    return [
        (tokenizer.int_to_str[i.item()], p.item())
        for p, i in zip(top_p, top_i)
    ]


@torch.no_grad()
def token_weighted_perplexity(model, tokenizer, device, dataset, batch_size):
    collator = A2DataCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    unk_count = 0
    valid_count = 0

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        output = model(input_ids=input_ids, labels=labels)
        shifted = labels[:, 1:]
        valid = (shifted != -100).sum().item()
        if valid == 0:
            continue
        total_nll += output.loss.item() * valid
        total_tokens += valid
        valid_count += valid
        unk_count += ((shifted == tokenizer.unk_token_id) & (shifted != -100)).sum().item()
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1} batches")

    ce = total_nll / total_tokens
    return ce, math.exp(ce), 100.0 * unk_count / valid_count if valid_count else 0.0


def main():
    device = select_device()
    print("=" * 60)
    print("A2 MODEL EVALUATION")
    print("=" * 60)
    print("Device:", device)
    print("Model dir:", MODEL_DIR)

    # --- Reload verification ---
    print("\n[1/4] Reload verification...")
    tokenizer = A1Tokenizer.from_file(TOKENIZER_FILE)
    model = A2Transformer.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    test = collator = A2DataCollator(tokenizer)
    batch = collator([{"text": "She lives in San"}])
    out = model(batch["input_ids"].to(device))
    assert out.logits.shape[-1] == len(tokenizer)
    assert torch.isfinite(out.logits).all()
    print("  Reload OK, logits shape:", tuple(out.logits.shape))

    # --- Colab metrics from experiment_info ---
    info_path = os.path.join(MODEL_DIR, "experiment_info.json")
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        print("\n[2/4] Colab Trainer metrics (from experiment_info.json):")
        print(f"  val cross-entropy: {info['trainer_validation_cross_entropy']:.4f}")
        print(f"  val perplexity:    {info['trainer_validation_perplexity']:.2f}")
        print(f"  parameters:        {info['num_parameters']:,}")

    # --- Next-word prediction ---
    print("\n[3/4] Next-word prediction:")
    for prompt in PROMPTS:
        print(f"\n  Prompt: {prompt!r}")
        for rank, (word, prob) in enumerate(
            predict_next_words(model, tokenizer, device, prompt, k=10),
            start=1,
        ):
            print(f"    {rank:2d}. {word:15s} {prob:.4f}")

    # --- Token-weighted validation perplexity ---
    print("\n[4/4] Token-weighted validation perplexity (full val set)...")
    dataset = load_dataset("text", data_files={"val": VAL_FILE})
    val_dataset = dataset["val"].filter(lambda x: x["text"].strip() != "")
    print("  Validation examples:", len(val_dataset))

    ce, ppl, unk_pct = token_weighted_perplexity(
        model, tokenizer, device, val_dataset, EVAL_BATCH_SIZE,
    )
    print(f"\n  Validation cross-entropy: {ce:.4f}")
    print(f"  Validation perplexity:    {ppl:.2f}")
    print(f"  Validation UNK %:         {unk_pct:.4f}%")

    results = {
        "token_weighted_val_cross_entropy": ce,
        "token_weighted_val_perplexity": ppl,
        "val_unk_pct": unk_pct,
    }
    if os.path.exists(info_path):
        results["trainer_val_cross_entropy"] = info["trainer_validation_cross_entropy"]
        results["trainer_val_perplexity"] = info["trainer_validation_perplexity"]

    out_path = os.path.join(MODEL_DIR, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
