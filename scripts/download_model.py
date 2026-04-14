#!/usr/bin/env python3
"""Download TinyStories-1M model files for zgml inference.

Requires: pip install huggingface_hub

Downloads model weights (safetensors), vocab.json, and merges.txt
to data/tinystories/.
"""

import os
from huggingface_hub import hf_hub_download

REPO_ID = "roneneldan/TinyStories-1M"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tinystories")

FILES = [
    "vocab.json",
    "merges.txt",
]

os.makedirs(OUT_DIR, exist_ok=True)

# Try safetensors first, fall back to pytorch_model.bin + convert
try:
    path = hf_hub_download(REPO_ID, "model.safetensors", local_dir=OUT_DIR)
    print(f"Downloaded model.safetensors -> {path}")
except Exception:
    print("No safetensors file found, downloading pytorch_model.bin and converting...")
    pt_path = hf_hub_download(REPO_ID, "pytorch_model.bin", local_dir=OUT_DIR)
    print(f"Downloaded pytorch_model.bin -> {pt_path}")

    # Convert to safetensors
    import torch
    from safetensors.torch import save_file
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    sf_path = os.path.join(OUT_DIR, "model.safetensors")
    save_file(state_dict, sf_path)
    print(f"Converted to {sf_path}")

for fname in FILES:
    path = hf_hub_download(REPO_ID, fname, local_dir=OUT_DIR)
    print(f"Downloaded {fname} -> {path}")

print(f"\nAll files saved to {OUT_DIR}/")
print("Run: ./zig-out/bin/generate-pretrained data/tinystories/model.safetensors "
      "data/tinystories/vocab.json data/tinystories/merges.txt \"Once upon a time\"")
