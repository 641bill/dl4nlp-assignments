import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

from A1_skeleton import (
    A1Tokenizer,
    A1RNNModel,
)


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = "trainer_output"
TOKENIZER_FILE = "trainer_output/tokenizer.pkl"
VAL_FILE = "val.txt"

EVAL_BATCH_SIZE = 32


# ============================================================
# 0. Select device
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


# ============================================================
# 1. Load trained model and tokenizer
# ============================================================

print("\nLoading model and tokenizer...")

tokenizer = A1Tokenizer.from_file(
    TOKENIZER_FILE
)

model = A1RNNModel.from_pretrained(
    MODEL_DIR
)

model.to(device)
model.eval()

print("Vocabulary size:", len(tokenizer))
print("Model loaded successfully.")


# ============================================================
# Task 5.1
# Predicting the next word
# ============================================================

def predict_next_words(prompt, k=10):
    """
    Return the k most likely next words after `prompt`.
    """

    encoding = tokenizer(
        [prompt],
        truncation=True,
        padding=False,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)

    with torch.no_grad():
        output = model(input_ids)

    # Our tokenizer normally creates:
    #
    # <BOS> she lives in san <EOS>
    #
    # The logits at "san" are therefore at position -2.
    #
    # If EOS was somehow absent due to truncation,
    # fall back to the final position.
    if input_ids[0, -1].item() == tokenizer.eos_token_id:
        prediction_position = -2
    else:
        prediction_position = -1

    next_token_logits = output.logits[
        0,
        prediction_position,
        :
    ]

    # Convert raw logits to probabilities.
    probabilities = torch.softmax(
        next_token_logits,
        dim=-1,
    )

    top_probabilities, top_ids = torch.topk(
        probabilities,
        k=k,
    )

    results = []

    for probability, token_id in zip(
        top_probabilities,
        top_ids,
    ):
        token_id = token_id.item()

        token = tokenizer.int_to_str[token_id]

        results.append(
            (token, probability.item())
        )

    return results


print("\n" + "=" * 60)
print("TASK 5.1: NEXT-WORD PREDICTION")
print("=" * 60)

prompts = [
    "She lives in San",
    "The capital of Sweden is",
    "The cat sat on the",
]

for prompt in prompts:
    print(f"\nPrompt: {prompt!r}")

    predictions = predict_next_words(
        prompt,
        k=10,
    )

    for rank, (word, probability) in enumerate(
        predictions,
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"{word:15s} "
            f"{probability:.4f}"
        )


# ============================================================
# Task 5.2
# Validation perplexity
# ============================================================

print("\n" + "=" * 60)
print("TASK 5.2: PERPLEXITY")
print("=" * 60)


dataset = load_dataset(
    "text",
    data_files={
        "val": VAL_FILE,
    },
)

val_dataset = dataset["val"].filter(
    lambda x: x["text"].strip() != ""
)

print(
    "Validation examples:",
    len(val_dataset),
)


def compute_perplexity(
    model,
    tokenizer,
    dataset,
    batch_size=32,
):
    """
    Compute token-weighted validation cross-entropy
    and perplexity.
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    total_negative_log_likelihood = 0.0
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

            # Labels initially equal the actual sequence.
            labels = input_ids.clone()

            # Padding is artificial, so exclude it
            # from the language-model loss.
            labels[
                labels == tokenizer.pad_token_id
            ] = -100

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            output = model(
                input_ids=input_ids,
                labels=labels,
            )

            # The model internally compares:
            #
            # logits[:, :-1]
            # labels[:, 1:]
            #
            # so count valid targets after that shift.
            shifted_labels = labels[:, 1:]

            valid_tokens = (
                shifted_labels != -100
            ).sum().item()

            if valid_tokens == 0:
                continue

            # output.loss is the MEAN cross-entropy
            # over valid tokens.
            #
            # Multiply by token count to recover the
            # total NLL for this batch.
            total_negative_log_likelihood += (
                output.loss.item()
                * valid_tokens
            )

            total_tokens += valid_tokens

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Processed "
                    f"{batch_index + 1} batches..."
                )

    average_loss = (
        total_negative_log_likelihood
        / total_tokens
    )

    perplexity = math.exp(
        average_loss
    )

    unk_count = 0
    valid_count = 0

    for batch in loader:
        encoding = tokenizer(
            batch["text"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"]
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        shifted_labels = labels[:, 1:]

        valid_mask = shifted_labels != -100

        valid_count += valid_mask.sum().item()

        unk_count += (
            (shifted_labels == tokenizer.unk_token_id)
            & valid_mask
        ).sum().item()

    print(
        "UNK percentage:",
        100 * unk_count / valid_count
    )

    return average_loss, perplexity


val_loss, perplexity = compute_perplexity(
    model,
    tokenizer,
    val_dataset,
    batch_size=EVAL_BATCH_SIZE,
)

print(f"\nValidation cross-entropy: {val_loss:.4f}")
print(f"Validation perplexity:    {perplexity:.2f}")


# ============================================================
# Task 5.3
# Inspect learned embeddings
# ============================================================

print("\n" + "=" * 60)
print("TASK 5.3: EMBEDDING NEIGHBORS")
print("=" * 60)


def nearest_neighbors(
    word,
    k=10,
):
    """
    Find words whose learned embedding vectors have
    the highest cosine similarity to `word`.
    """

    # Our tokenizer lowercases ordinary text.
    word = word.lower()

    if word not in tokenizer.str_to_int:
        return None

    word_id = tokenizer.str_to_int[word]

    # Shape:
    #
    # (vocab_size, embedding_size)
    embeddings = (
        model.embedding.weight
        .detach()
        .cpu()
    )

    # Normalize every vector to length 1.
    #
    # After normalization, dot product =
    # cosine similarity.
    normalized_embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    query_vector = normalized_embeddings[
        word_id
    ]

    # Compare query vector against every
    # vocabulary embedding.
    similarities = (
        normalized_embeddings
        @ query_vector
    )

    # Otherwise the nearest word would trivially
    # be the word itself.
    similarities[word_id] = float("-inf")

    # Special tokens are not interesting here.
    special_tokens = [
        tokenizer.pad_token,
        tokenizer.unk_token,
        tokenizer.bos_token,
        tokenizer.eos_token,
    ]

    for token in special_tokens:
        token_id = tokenizer.str_to_int[token]
        similarities[token_id] = float("-inf")

    scores, ids = torch.topk(
        similarities,
        k=k,
    )

    results = []

    for score, token_id in zip(
        scores,
        ids,
    ):
        token_id = token_id.item()

        neighbor = tokenizer.int_to_str[
            token_id
        ]

        results.append(
            (neighbor, score.item())
        )

    return results


words_to_inspect = [
    "sweden",
    "stockholm",
    "city",
    "computer",
    "king",
]

for word in words_to_inspect:

    print(f"\nNearest neighbors of {word!r}:")

    neighbors = nearest_neighbors(
        word,
        k=10,
    )

    if neighbors is None:
        print("  Word is not in the vocabulary.")
        continue

    for rank, (neighbor, similarity) in enumerate(
        neighbors,
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"{neighbor:15s} "
            f"{similarity:.3f}"
        )


print("\n" + "=" * 60)
print("PART 5 COMPLETE")
print("=" * 60)

