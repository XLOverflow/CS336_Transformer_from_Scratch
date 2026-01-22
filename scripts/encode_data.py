#!/usr/bin/env python3

"""
Data encoding script for TinyStories and OpenWebText datasets into BPE token IDs.
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.tokenizers.tokenizer import BPETokenizer

# OWT 数据
train_data_path = "/ocean/projects/cis250265p/xli45/assignment1-basics/data/owt_train.txt"
valid_data_path = "/ocean/projects/cis250265p/xli45/assignment1-basics/data/owt_valid.txt"
output_dir = "/ocean/projects/cis250265p/xli45/assignment1-basics/data"
tokenizer_dir = "/ocean/projects/cis250265p/xli45/assignment1-basics/trained_tokenizers/owt"

def main():
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.from_files(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt"),
        special_tokens=["<|endoftext|>"]
    )

    # Training data
    train_output = os.path.join(output_dir, "owt_train_encoded.npy")

    if not os.path.exists(train_output):
        print("Loading training data...")
        with open(train_data_path, "r", encoding="utf-8") as f:
            train_data = f.read()
        print(f"Training data size: {len(train_data):,} characters")

        print("Encoding training data in parallel...")
        tokenizer.encode_parallel(
            train_data,
            num_processes=10,
            split_token="<|endoftext|>",
            output_file=train_output
        )
        del train_data
    else:
        print(f"Training data already exists: {train_output}")

    train_arr = np.memmap(train_output, dtype=np.uint16, mode='r')
    print(f"Training tokens: {len(train_arr):,}")
    del train_arr

    # Validation data
    valid_output = os.path.join(output_dir, "owt_valid_encoded.npy")

    if not os.path.exists(valid_output):
        print("Loading validation data...")
        with open(valid_data_path, "r", encoding="utf-8") as f:
            valid_data = f.read()
        print(f"Validation data size: {len(valid_data):,} characters")

        print("Encoding validation data in parallel...")
        tokenizer.encode_parallel(
            valid_data,
            num_processes=10,
            split_token="<|endoftext|>",
            output_file=valid_output
        )
        del valid_data
    else:
        print(f"Validation data already exists: {valid_output}")

    valid_arr = np.memmap(valid_output, dtype=np.uint16, mode='r')
    print(f"Validation tokens: {len(valid_arr):,}")
    del valid_arr

    print("Done!")

if __name__ == "__main__":
    main()