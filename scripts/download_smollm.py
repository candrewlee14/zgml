#!/usr/bin/env python3
"""Download SmolLM-135M weights for zgml tests and benchmarks.

This intentionally does not create tokenizer files: zgml exposes a token-level
LLaMA session API, not a text generation CLI.

Usage:
    python scripts/download_smollm.py [output_dir]
    # Default output: data/smollm/
"""

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

    print(f"Downloading {REPO_ID} model.safetensors...")
    model_path = hf_hub_download(REPO_ID, "model.safetensors", local_dir=OUTPUT_DIR)
    print(f"  -> {model_path}")

    print("\nDone! Use token IDs through zgml.llm.LlamaSession, or run the SmolLM benchmark.")


if __name__ == "__main__":
    main()
