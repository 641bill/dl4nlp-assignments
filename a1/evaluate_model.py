"""Evaluate a trained RNN language model (Tasks 5.1–5.3).

Usage:
    python evaluate_model.py trainer_output_50k
    python evaluate_model.py trainer_output_10k
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

from A1_skeleton import A1Tokenizer, A1RNNModel

VAL_FILE = "val.txt"
EVAL_BATCH_SIZE = 32

PROMPTS = [
    "She lives in San",
    "The capital of Sweden is",
    "The cat sat on the",
]

NEIGHBOR_WORDS = [
    "sweden",
    "stockholm",
    "city",
    "computer",
    "king",
]


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

    if input_ids[0, -1].item() == tokenizer.eos_token_id:
        prediction_position = -2
    else:
        prediction_position = -1

    next_token_logits = output.logits[0, prediction_position, :]
    probabilities = torch.softmax(next_token_logits, dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, k=k)

    results = []
    for probability, token_id in zip(top_probabilities, top_ids):
        token = tokenizer.int_to_str[token_id.item()]
        results.append((token, probability.item()))
    return results


def compute_perplexity(model, tokenizer, device, dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            encoding = tokenizer(
                batch["text"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"]
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            output = model(input_ids=input_ids, labels=labels)
            shifted_labels = labels[:, 1:]
            valid_tokens = (shifted_labels != -100).sum().item()
            if valid_tokens == 0:
                continue

            total_nll += output.loss.item() * valid_tokens
            total_tokens += valid_tokens

            if (batch_index + 1) % 100 == 0:
                print(f"  Processed {batch_index + 1} batches...")

    average_loss = total_nll / total_tokens
    perplexity = math.exp(average_loss)

    unk_count = 0
    valid_count = 0
    for batch in loader:
        encoding = tokenizer(
            batch["text"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = encoding["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        shifted_labels = labels[:, 1:]
        valid_mask = shifted_labels != -100
        valid_count += valid_mask.sum().item()
        unk_count += (
            (shifted_labels == tokenizer.unk_token_id) & valid_mask
        ).sum().item()

    unk_pct = 100.0 * unk_count / valid_count if valid_count else 0.0
    return average_loss, perplexity, unk_pct


def nearest_neighbors(model, tokenizer, word, k=10):
    word = word.lower()
    if word not in tokenizer.str_to_int:
        return None

    word_id = tokenizer.str_to_int[word]
    embeddings = model.embedding.weight.detach().cpu()
    normalized = F.normalize(embeddings, p=2, dim=1)
    query = normalized[word_id]
    similarities = normalized @ query
    similarities[word_id] = float("-inf")

    for token in [
        tokenizer.pad_token,
        tokenizer.unk_token,
        tokenizer.bos_token,
        tokenizer.eos_token,
    ]:
        similarities[tokenizer.str_to_int[token]] = float("-inf")

    scores, ids = torch.topk(similarities, k=k)
    return [
        (tokenizer.int_to_str[token_id.item()], score.item())
        for score, token_id in zip(scores, ids)
    ]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def model_size_on_disk(model_dir):
    total = 0
    for name in os.listdir(model_dir):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            total += os.path.getsize(path)
    return total


def main():
    parser = argparse.ArgumentParser(description="Evaluate an A1 RNN language model.")
    parser.add_argument(
        "model_dir",
        help="Directory containing model weights and tokenizer.pkl",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    tokenizer_file = os.path.join(model_dir, "tokenizer.pkl")

    if not os.path.isdir(model_dir):
        sys.exit(f"Model directory not found: {model_dir}")
    if not os.path.exists(tokenizer_file):
        sys.exit(f"Tokenizer not found: {tokenizer_file}")

    device = select_device()
    print("=" * 60)
    print(f"EVALUATION: {model_dir}")
    print("=" * 60)
    print("Device:", device)

    tokenizer = A1Tokenizer.from_file(tokenizer_file)
    model = A1RNNModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    print("Vocabulary size:", len(tokenizer))
    print("Parameters:", f"{count_parameters(model):,}")
    print("Model size on disk:", f"{model_size_on_disk(model_dir) / 1e6:.2f} MB")

    # Task 5.1
    print("\n" + "=" * 60)
    print("TASK 5.1: NEXT-WORD PREDICTION")
    print("=" * 60)
    prompt_results = {}
    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt!r}")
        predictions = predict_next_words(model, tokenizer, device, prompt, k=10)
        prompt_results[prompt] = predictions
        for rank, (word, prob) in enumerate(predictions, start=1):
            print(f"{rank:2d}. {word:15s} {prob:.4f}")

    # Task 5.2
    print("\n" + "=" * 60)
    print("TASK 5.2: PERPLEXITY")
    print("=" * 60)
    dataset = load_dataset("text", data_files={"val": VAL_FILE})
    val_dataset = dataset["val"].filter(lambda x: x["text"].strip() != "")
    print("Validation examples:", len(val_dataset))

    val_loss, perplexity, unk_pct = compute_perplexity(
        model, tokenizer, device, val_dataset, batch_size=EVAL_BATCH_SIZE,
    )
    print(f"\nValidation cross-entropy: {val_loss:.4f}")
    print(f"Validation perplexity:    {perplexity:.2f}")
    print(f"Validation UNK %:         {unk_pct:.4f}%")

    # Task 5.3
    print("\n" + "=" * 60)
    print("TASK 5.3: EMBEDDING NEIGHBORS")
    print("=" * 60)
    for word in NEIGHBOR_WORDS:
        print(f"\nNearest neighbors of {word!r}:")
        neighbors = nearest_neighbors(model, tokenizer, word, k=10)
        if neighbors is None:
            print("  Word is not in the vocabulary.")
            continue
        for rank, (neighbor, sim) in enumerate(neighbors, start=1):
            print(f"{rank:2d}. {neighbor:15s} {sim:.3f}")

    # Write results summary
    results_path = os.path.join(model_dir, "evaluation_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation results for {model_dir}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"vocabulary_size = {len(tokenizer)}\n")
        f.write(f"parameters      = {count_parameters(model):,}\n")
        f.write(f"disk_size_bytes = {model_size_on_disk(model_dir)}\n")
        f.write(f"val_cross_entropy = {val_loss:.4f}\n")
        f.write(f"val_perplexity    = {perplexity:.2f}\n")
        f.write(f"val_unk_pct       = {unk_pct:.4f}%\n\n")
        for prompt, preds in prompt_results.items():
            f.write(f"Prompt: {prompt!r}\n")
            for rank, (word, prob) in enumerate(preds, start=1):
                f.write(f"  {rank:2d}. {word:15s} {prob:.4f}\n")
            f.write("\n")
    print(f"\nResults saved to: {results_path}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
