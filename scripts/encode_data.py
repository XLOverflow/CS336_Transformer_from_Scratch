#!/usr/bin/env python3

"""
Data encoding script for TinyStories and OpenWebText datasets into BPE token IDs.

Usage:
    python scripts/encode_data.py --dataset tinystories
    python scripts/encode_data.py --dataset owt
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.tokenizers.tokenizer import BPETokenizer

BASE = "/ocean/projects/cis250265p/xli45/assignment1-basics"

DATASETS = {
    "tinystories": {
        "train_data": f"{BASE}/data/TinyStoriesV2-GPT4-train.txt",
        "valid_data": f"{BASE}/data/TinyStoriesV2-GPT4-valid.txt",
        "tokenizer_dir": f"{BASE}/trained_tokenizers/tinystories",
        "train_output": f"{BASE}/data/tinystories_train_encoded.npy",
        "valid_output": f"{BASE}/data/tinystories_valid_encoded.npy",
    },
    "owt": {
        "train_data": f"{BASE}/data/owt_train.txt",
        "valid_data": f"{BASE}/data/owt_valid.txt",
        "tokenizer_dir": f"{BASE}/trained_tokenizers/owt",
        "train_output": f"{BASE}/data/owt_train_encoded.npy",
        "valid_output": f"{BASE}/data/owt_valid_encoded.npy",
    },
}


def encode_file(tokenizer, input_path, output_path, num_processes=10):
    """Encode a single text file to token IDs."""
    if os.path.exists(output_path):
        print(f"Already exists: {output_path}")
        arr = np.memmap(output_path, dtype=np.uint16, mode='r')
        print(f"  Tokens: {len(arr):,}")
        del arr
        return

    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  Characters: {len(text):,}")

    print("Encoding in parallel...")
    tokenizer.encode_parallel(
        text,
        num_processes=num_processes,
        split_token="<|endoftext|>",
        output_file=output_path
    )
    del text

    arr = np.memmap(output_path, dtype=np.uint16, mode='r')
    print(f"  Tokens: {len(arr):,}")
    del arr


def main():
    parser = argparse.ArgumentParser(description="Encode text data into BPE token IDs")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["tinystories", "owt"],
                        help="Dataset to encode")
    parser.add_argument("--num_processes", type=int, default=10)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]

    print(f"=== Encoding {args.dataset} ===")
    print(f"Loading tokenizer from {cfg['tokenizer_dir']}...")
    tokenizer = BPETokenizer.from_files(
        os.path.join(cfg["tokenizer_dir"], "vocab.json"),
        os.path.join(cfg["tokenizer_dir"], "merges.txt"),
        special_tokens=["<|endoftext|>"]
    )
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # Training data
    encode_file(tokenizer, cfg["train_data"], cfg["train_output"], args.num_processes)

    # Validation data
    encode_file(tokenizer, cfg["valid_data"], cfg["valid_output"], args.num_processes)

    print("Done!")


if __name__ == "__main__":
    main()
