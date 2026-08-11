"""Compare preserved 10k and newly trained 50k experiments."""

import os

from A1_skeleton import A1Tokenizer, A1RNNModel

DIR_10K = "trainer_output_10k"
DIR_50K = "trainer_output_50k"
REPORT_FILE = "experiment_comparison.txt"

# Known 10k results from original evaluation.
KNOWN_10K = {
    "val_cross_entropy": 4.3219,
    "val_perplexity": 75.33,
    "val_unk_pct": 11.2424,
    "san_prompt_top": "francisco ~ 0.474, diego ~ 0.216, <UNK> ~ 0.161",
}


def read_eval_results(model_dir):
    path = os.path.join(model_dir, "evaluation_results.txt")
    if not os.path.exists(path):
        return None
    return open(path, encoding="utf-8").read()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def dir_size(model_dir):
    return sum(
        os.path.getsize(os.path.join(model_dir, f))
        for f in os.listdir(model_dir)
        if os.path.isfile(os.path.join(model_dir, f))
    )


def load_metrics(model_dir, known=None):
    tok = A1Tokenizer.from_file(os.path.join(model_dir, "tokenizer.pkl"))
    model = A1RNNModel.from_pretrained(model_dir)
    metrics = {
        "vocab_size": len(tok),
        "parameters": count_parameters(model),
        "disk_bytes": dir_size(model_dir),
    }

    eval_path = os.path.join(model_dir, "evaluation_results.txt")
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("val_cross_entropy"):
                    metrics["val_cross_entropy"] = float(line.split("=")[1])
                elif line.startswith("val_perplexity"):
                    metrics["val_perplexity"] = float(line.split("=")[1])
                elif line.startswith("val_unk_pct"):
                    metrics["val_unk_pct"] = float(line.split("=")[1].rstrip("%"))
    elif known:
        metrics.update(known)

    return metrics


def fmt(val, is_pct=False):
    if val is None:
        return "N/A"
    if is_pct:
        return f"{val:.4f}%"
    if isinstance(val, float):
        return f"{val:.4f}" if val < 100 else f"{val:.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def main():
    if not os.path.isdir(DIR_10K):
        raise FileNotFoundError(f"Missing {DIR_10K}/")
    if not os.path.isdir(DIR_50K):
        raise FileNotFoundError(
            f"Missing {DIR_50K}/. Run part4.py and evaluate_model.py first."
        )

    m10 = load_metrics(DIR_10K, known=KNOWN_10K)
    m50 = load_metrics(DIR_50K)

    lines = []
    lines.append("10k vs 50k Vocabulary Experiment Comparison")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'Metric':<30} {'10k':>16} {'up-to-50k':>16}")
    lines.append("-" * 62)
    lines.append(
        f"{'Actual vocabulary size':<30} {fmt(m10['vocab_size']):>16} {fmt(m50['vocab_size']):>16}"
    )
    lines.append(
        f"{'Validation UNK %':<30} {fmt(m10.get('val_unk_pct'), True):>16} {fmt(m50.get('val_unk_pct'), True):>16}"
    )
    lines.append(
        f"{'Validation cross-entropy':<30} {fmt(m10.get('val_cross_entropy')):>16} {fmt(m50.get('val_cross_entropy')):>16}"
    )
    lines.append(
        f"{'Validation perplexity':<30} {fmt(m10.get('val_perplexity')):>16} {fmt(m50.get('val_perplexity')):>16}"
    )
    lines.append(
        f"{'Parameter count':<30} {fmt(m10['parameters']):>16} {fmt(m50['parameters']):>16}"
    )
    lines.append(
        f"{'Model size on disk (MB)':<30} {m10['disk_bytes']/1e6:>16.2f} {m50['disk_bytes']/1e6:>16.2f}"
    )
    lines.append("")
    lines.append('Qualitative: "She lives in San" top predictions')
    lines.append(f"  10k:      {KNOWN_10K['san_prompt_top']}")
    if os.path.exists(os.path.join(DIR_50K, "evaluation_results.txt")):
        with open(os.path.join(DIR_50K, "evaluation_results.txt"), encoding="utf-8") as f:
            text = f.read()
        for line in text.splitlines():
            if line.startswith("  1.") and "Prompt" not in line:
                # grab first prediction after She lives in San block
                pass
        # simpler: read from eval file
        in_san = False
        san_preds = []
        for line in text.splitlines():
            if line.strip() == "Prompt: 'She lives in San'":
                in_san = True
                continue
            if in_san and line.startswith("  1."):
                san_preds = []
            if in_san and line.startswith("  "):
                san_preds.append(line.strip())
                if len(san_preds) >= 3:
                    break
            if in_san and line.startswith("Prompt:") and "She lives" not in line:
                break
        if san_preds:
            lines.append(f"  up-to-50k: {', '.join(san_preds[:3])}")

    lines.append("")
    lines.append("Interpretation")
    lines.append("-" * 60)
    lines.append(
        "A larger vocabulary does NOT automatically mean lower perplexity. "
        "Perplexity measures how well the model predicts the exact next token "
        "among all vocabulary items. When rare words that previously mapped to "
        "<UNK> become separate target classes, the prediction problem becomes "
        "harder: the model must distinguish among many low-frequency tokens "
        "instead of lumping them into one UNK bucket. Lower UNK% and higher "
        "perplexity can both indicate a harder but more informative evaluation."
    )

    report = "\n".join(lines)
    print(report)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
