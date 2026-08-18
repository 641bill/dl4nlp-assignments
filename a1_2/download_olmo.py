"""Download allenai/OLMo-2-0425-1B with progress and verify both shards exist."""

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODEL = "allenai/OLMo-2-0425-1B"


def verify(path: Path) -> bool:
    idx = json.loads((path / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx["weight_map"].values()))
    ok = True
    for shard in shards:
        p = path / shard
        if p.exists():
            print(f"  OK   {shard} ({p.resolve().stat().st_size / 1e9:.2f} GB)")
        else:
            print(f"  MISS {shard}")
            ok = False
    return ok


def main():
    print(f"Downloading {MODEL} (resumes if partially cached)...", flush=True)
    path = Path(snapshot_download(MODEL, resume_download=True))
    print(f"\nSnapshot: {path}", flush=True)
    print("Checking weight shards:", flush=True)
    if verify(path):
        print("\nDownload complete. Run Task 3.3 with:")
        print("  .venv/bin/python a1_2/generate_a2.py --only-olmo --device cpu")
        return 0
    print("\nERROR: weight shards still missing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
