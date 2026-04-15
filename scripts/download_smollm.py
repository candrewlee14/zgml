#!/usr/bin/env python3
"""Download SmolLM-135M and extract tokenizer files for zgml.

Downloads the safetensors model and converts the HF tokenizer.json
into separate vocab.json and merges.txt files compatible with zgml's
GPT2Tokenizer.

Usage:
    python scripts/download_smollm.py [output_dir]
    # Default output: data/smollm/
"""

import json
import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("pip install huggingface-hub", file=sys.stderr)
    sys.exit(1)


REPO_ID = "HuggingFaceTB/SmolLM-135M"
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/smollm"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download model weights.
    print(f"Downloading {REPO_ID} model.safetensors...")
    model_path = hf_hub_download(REPO_ID, "model.safetensors", local_dir=OUTPUT_DIR)
    print(f"  -> {model_path}")

    # Download tokenizer.json and extract vocab + merges.
    print(f"Downloading {REPO_ID} tokenizer.json...")
    tok_path = hf_hub_download(REPO_ID, "tokenizer.json", local_dir=OUTPUT_DIR)

    with open(tok_path) as f:
        tok = json.load(f)

    # Extract vocab: {token_string: id}
    vocab = tok["model"]["vocab"]
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"  -> {vocab_path} ({len(vocab)} tokens)")

    # Extract merges: list of "token1 token2" strings.
    merges = tok["model"]["merges"]
    merges_path = os.path.join(OUTPUT_DIR, "merges.txt")
    with open(merges_path, "w") as f:
        f.write("#version: 0.2\n")
        for merge in merges:
            f.write(merge + "\n")
    print(f"  -> {merges_path} ({len(merges)} merges)")

    print(f"\nDone! Run with:")
    print(f"  zig build generate-llama && ./zig-out/bin/generate-llama \\")
    print(f"    {OUTPUT_DIR}/model.safetensors {OUTPUT_DIR}/vocab.json {OUTPUT_DIR}/merges.txt \"Once upon a time\"")


if __name__ == "__main__":
    main()
