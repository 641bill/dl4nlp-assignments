"""Assignment 2 Step 3: next-word prediction and text generation."""

import argparse
import os
import sys
from datetime import datetime

import torch
from torch.distributions import Categorical

A1_DIR = os.path.join(os.path.dirname(__file__), "..", "a1")
sys.path.insert(0, A1_DIR)

from A1_skeleton import A1Tokenizer
from A2_skeleton import A2Transformer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "final_model")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.pkl")
OLMO_MODEL_NAME = "allenai/OLMo-2-0425-1B"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "compare_olmo_a2.txt")

NEXT_WORD_PROMPTS = [
    "She lives in San",
    "he lives in san",
    "The capital of Sweden is",
    "The cat sat on the",
]

GENERATION_PROMPTS = [
    "In natural language processing, a Transformer",
    "Is Stockholm the capital of Sweden? Answer yes or no. The answer is",
    "Write a Python program that reverses a list.",
    "She lives in San",
]


def select_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(device):
    if device.type in ("cuda", "mps"):
        return torch.float16
    return torch.float32


class Tee:
    """Write stdout to both console and a file."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)

    def flush(self):
        for f in self.files:
            f.flush()


def encode_prompt(tokenizer, prompt, continue_generation=True):
    encoding = tokenizer(
        [prompt],
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"]
    if continue_generation and input_ids[0, -1].item() == tokenizer.eos_token_id:
        input_ids = input_ids[:, :-1]
    return input_ids


def decode_ids(tokenizer, ids):
    tokens = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == tokenizer.bos_token_id:
            continue
        if token_id == tokenizer.eos_token_id:
            break
        if token_id == tokenizer.pad_token_id:
            continue
        tokens.append(tokenizer.int_to_str[token_id])
    return " ".join(tokens)


def last_content_position(input_ids, tokenizer):
    last = input_ids[0, -1].item()
    if last == tokenizer.eos_token_id and input_ids.shape[1] >= 2:
        return -2
    return -1


def predict_next_word(model, tokenizer, prompt, k=10, device=None):
    """Task 3.1: score the next word after a prompt (skip trailing EOS)."""
    if device is None:
        device = next(model.parameters()).device
    input_ids = encode_prompt(tokenizer, prompt, continue_generation=False).to(device)
    pos = last_content_position(input_ids, tokenizer)
    with torch.no_grad():
        logits = model(input_ids).logits[0, pos, :]
    probs = torch.softmax(logits, dim=-1)
    top_p, top_i = torch.topk(probs, k=min(k, probs.numel()))
    ranked = [
        (tokenizer.int_to_str[i.item()], p.item())
        for p, i in zip(top_p, top_i)
    ]
    greedy_id = int(torch.argmax(logits))
    return tokenizer.int_to_str[greedy_id], ranked


def apply_top_k(logits, topk):
    if topk is None or topk <= 0 or topk >= logits.numel():
        return logits
    values, indices = torch.topk(logits, k=topk)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, indices, values)
    return filtered


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=40,
    temperature=1.0,
    topk=None,
    device=None,
):
    """Task 3.2: sample tokens until EOS or max_length steps.

    temperature scales logits before sampling. topk keeps only the k most
    probable next tokens. temperature <= 0 is greedy (argmax).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    input_ids = encode_prompt(tokenizer, prompt, continue_generation=True).to(device)
    max_context = getattr(model.config, "max_position_embeddings", None)
    generated = []

    with torch.no_grad():
        for _ in range(max_length):
            context = input_ids
            if max_context is not None and context.shape[1] > max_context:
                context = context[:, -max_context:]
            logits = model(context).logits[0, -1, :].float()

            if temperature is None or temperature <= 0:
                next_id = int(torch.argmax(logits))
            else:
                logits = apply_top_k(logits / temperature, topk)
                next_id = int(Categorical(logits=logits).sample())

            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=device)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            if next_id == tokenizer.eos_token_id:
                break

    return decode_ids(tokenizer, input_ids[0].tolist())


def run_our_model(device, max_length, seed):
    print("=" * 60)
    print("TASK 3 — our trained A2 Transformer")
    print("=" * 60)
    print("Device:", device)
    print("Model:", MODEL_DIR)

    tokenizer = A1Tokenizer.from_file(TOKENIZER_FILE)
    model = A2Transformer.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    print("\n--- Task 3.1: next-word prediction ---")
    for prompt in NEXT_WORD_PROMPTS:
        greedy, ranked = predict_next_word(model, tokenizer, prompt, k=8, device=device)
        print(f"\nPrompt: {prompt!r}")
        print(f"  argmax: {greedy}")
        for rank, (word, prob) in enumerate(ranked, start=1):
            print(f"  {rank:2d}. {word:15s} {prob:.4f}")

    settings = [
        {"temperature": 0.0, "topk": None, "label": "greedy"},
        {"temperature": 0.7, "topk": 10, "label": "T=0.7, top-k=10"},
        {"temperature": 1.0, "topk": 50, "label": "T=1.0, top-k=50"},
        {"temperature": 1.2, "topk": 20, "label": "T=1.2, top-k=20"},
    ]

    print("\n--- Task 3.2: sampled generation ---")
    for prompt in GENERATION_PROMPTS:
        print(f"\nPrompt: {prompt!r}")
        for setting in settings:
            if seed is not None:
                torch.manual_seed(seed)
            text = generate_text(
                model,
                tokenizer,
                prompt,
                max_length=max_length,
                temperature=setting["temperature"],
                topk=setting["topk"],
                device=device,
            )
            print(f"  [{setting['label']}]")
            print(f"    {text}")


def log(msg):
    print(msg, flush=True)


def run_olmo(device, max_length, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("\n" + "=" * 60)
    log("TASK 3.3 — pretrained OLMo 2 1B")
    log("=" * 60)
    log(f"Loading {OLMO_MODEL_NAME}")
    log(f"Device: {device}")

    dtype = select_dtype(device)
    log("Step 1/3: loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        OLMO_MODEL_NAME, local_files_only=True
    )
    log("Step 2/3: loading model weights from local cache...")
    model = AutoModelForCausalLM.from_pretrained(
        OLMO_MODEL_NAME,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    log(f"Step 3/3: moving model to {device}...")
    model = model.to(device)
    model.eval()
    log("Model ready.")

    def olmo_predict_next_word(prompt, k=8):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(probs, k=min(k, probs.numel()))
        ranked = [
            (tokenizer.decode([i.item()]).strip(), p.item())
            for p, i in zip(top_p, top_i)
        ]
        greedy = tokenizer.decode([int(torch.argmax(logits))]).strip()
        return greedy, ranked

    def olmo_generate(prompt, temperature, topk):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        do_sample = temperature is not None and temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_length,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            if topk:
                gen_kwargs["top_k"] = topk
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, prompt_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    log("\n--- Task 3.1-style next-token prediction (OLMo tokenizer) ---")
    for prompt in NEXT_WORD_PROMPTS:
        log(f"  predicting: {prompt!r}")
        greedy, ranked = olmo_predict_next_word(prompt, k=8)
        log(f"\nPrompt: {prompt!r}")
        log(f"  argmax: {greedy!r}")
        for rank, (token, prob) in enumerate(ranked, start=1):
            display = token.replace("\n", "\\n")
            log(f"  {rank:2d}. {display:20s} {prob:.4f}")

    settings = [
        {"temperature": 0.0, "topk": None, "label": "greedy"},
        {"temperature": 0.7, "topk": 10, "label": "T=0.7, top-k=10"},
        {"temperature": 1.0, "topk": 50, "label": "T=1.0, top-k=50"},
    ]

    log("\n--- Task 3.2-style generation (assignment prompts) ---")
    for prompt in GENERATION_PROMPTS:
        log(f"\nPrompt: {prompt!r}")
        for setting in settings:
            if seed is not None:
                torch.manual_seed(seed)
            log(f"  generating [{setting['label']}]...")
            text = olmo_generate(prompt, setting["temperature"], setting["topk"])
            log(f"  [{setting['label']}]")
            log(f"    {text}")

    log("\n--- Notes for Task 3.3 ---")
    log("- OLMo is a base LM (not instruction-tuned), but much larger and better trained.")
    log("- Expect more coherent continuations than the homework-scale A2 model.")
    log("- Compare especially on factual prompts and code-style prompts.")


def parse_args():
    parser = argparse.ArgumentParser(description="A2 Step 3 text generation")
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare-olmo",
        action="store_true",
        help="Also run Task 3.3 with allenai/OLMo-2-0425-1B",
    )
    parser.add_argument(
        "--only-olmo",
        action="store_true",
        help="Run only Task 3.3 (skip our trained A2 model)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Save console output to this text file",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device for OLMo run (e.g. cpu, mps, cuda). Default: auto",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    if args.only_olmo and device.type == "mps" and args.device is None:
        # CPU is more reliable for first-time 1B weight download + load on Mac.
        device = torch.device("cpu")
        log("Using CPU for OLMo (more reliable than MPS for 1B load).")
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(f"A2 Step 3 comparison\nGenerated: {datetime.now().isoformat()}\n\n")
        old_stdout = sys.stdout
        sys.stdout = Tee(old_stdout, out_file)
        try:
            if not args.only_olmo:
                run_our_model(device, args.max_length, args.seed)
            if args.compare_olmo or args.only_olmo:
                run_olmo(device, args.max_length, args.seed)
            print("\nDone.")
            print(f"Saved output to: {output_path}")
        finally:
            sys.stdout = old_stdout

    print(f"Task 3 output saved to {output_path}")


if __name__ == "__main__":
    main()
