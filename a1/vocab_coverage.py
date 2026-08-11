"""Analyze training-corpus token coverage for different vocabulary limits."""

import nltk
from collections import Counter

from A1_skeleton import lowercase_tokenizer, build_tokenizer

TRAIN_FILE = "train.txt"
SPECIAL_TOKEN_COUNT = 4
LIMITS = [10_000, 20_000, 30_000, 40_000, 50_000]


def main():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    counter = Counter()
    total_occurrences = 0

    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = lowercase_tokenizer(line)
            counter.update(tokens)
            total_occurrences += len(tokens)

    unique_types = len(counter)

    print("Training corpus statistics")
    print("=" * 60)
    print(f"Total token occurrences: {total_occurrences:,}")
    print(f"Unique token types:      {unique_types:,}")
    print()

    header = (
        f"{'Vocab limit':>12} | {'Actual vocab':>12} | "
        f"{'Token coverage':>14} | {'Estimated UNK %':>15}"
    )
    print(header)
    print("-" * len(header))

    for limit in LIMITS:
        tokenizer = build_tokenizer(
            train_file=TRAIN_FILE,
            max_voc_size=limit,
        )
        actual_vocab = len(tokenizer)
        in_vocab = set(tokenizer.str_to_int) - {
            tokenizer.pad_token,
            tokenizer.unk_token,
            tokenizer.bos_token,
            tokenizer.eos_token,
        }

        covered = sum(
            count for token, count in counter.items() if token in in_vocab
        )
        coverage = 100.0 * covered / total_occurrences
        unk_pct = 100.0 - coverage

        print(
            f"{limit:>12,} | {actual_vocab:>12,} | "
            f"{coverage:>13.4f}% | {unk_pct:>14.4f}%"
        )

    max_ordinary = LIMITS[-1] - SPECIAL_TOKEN_COUNT
    if unique_types <= max_ordinary:
        print()
        print(
            f"Note: corpus has {unique_types:,} unique ordinary tokens, "
            f"which fits within the {LIMITS[-1]:,} limit "
            f"({max_ordinary:,} ordinary slots)."
        )


if __name__ == "__main__":
    main()
