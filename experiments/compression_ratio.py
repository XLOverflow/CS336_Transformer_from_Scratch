#!/usr/bin/env python3
"""
Compression Ratio Experiment

Samples 10 documents from TinyStories and OpenWebText datasets,
encodes them using their respective trained tokenizers, and
calculates the compression ratio (bytes/token) for each.

TinyStories: 10K vocabulary tokenizer
OpenWebText: 32K vocabulary tokenizer
"""

import sys
import random
import time
from pathlib import Path

# Add parent directory to path to import cs336_basics
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.tokenizers.tokenizer import BPETokenizer


def sample_documents(file_path: str, num_samples: int = 10, delimiter: str = "<|endoftext|>") -> list[str]:
    """
    Sample documents from a text file using reservoir sampling (memory efficient).
    Documents are separated by the delimiter (e.g., <|endoftext|>).

    Args:
        file_path: Path to the text file
        num_samples: Number of documents to sample
        delimiter: Document separator

    Returns:
        List of sampled document strings
    """
    print(f"Reading from {file_path}...")

    # Use reservoir sampling to avoid loading entire file
    random.seed(42)  # For reproducibility

    reservoir = []
    doc_count = 0
    current_doc = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if delimiter in line:
                # Split the line by delimiter
                parts = line.split(delimiter)
                for i, part in enumerate(parts):
                    if i == 0:
                        # First part belongs to current doc
                        current_doc.append(part)
                        doc_text = ''.join(current_doc).strip()
                        if doc_text:
                            doc_count += 1
                            # Reservoir sampling
                            if len(reservoir) < num_samples:
                                reservoir.append(doc_text)
                            else:
                                j = random.randint(0, doc_count - 1)
                                if j < num_samples:
                                    reservoir[j] = doc_text
                        current_doc = []
                    else:
                        # Subsequent parts start new docs
                        if part.strip():
                            current_doc = [part]
            else:
                current_doc.append(line)

    # Don't forget the last document
    doc_text = ''.join(current_doc).strip()
    if doc_text:
        doc_count += 1
        if len(reservoir) < num_samples:
            reservoir.append(doc_text)
        else:
            j = random.randint(0, doc_count - 1)
            if j < num_samples:
                reservoir[j] = doc_text

    print(f"Total documents processed: {doc_count}")
    print(f"Sampled {len(reservoir)} documents")

    return reservoir


def calculate_compression_ratio(tokenizer: BPETokenizer, documents: list[str]) -> dict:
    """
    Calculate compression ratio (bytes/token) for a list of documents.

    Args:
        tokenizer: BPE tokenizer to use
        documents: List of document strings

    Returns:
        Dictionary with compression statistics
    """
    total_bytes = 0
    total_tokens = 0

    doc_stats = []
    
    # Timing
    start_time = time.time()

    for i, doc in enumerate(documents):
        # Get byte count (UTF-8 encoding)
        byte_count = len(doc.encode('utf-8'))

        # Encode to get token count
        token_ids = tokenizer.encode(doc)
        token_count = len(token_ids)

        # Calculate per-document ratio
        ratio = byte_count / token_count if token_count > 0 else 0

        doc_stats.append({
            'doc_index': i + 1,
            'bytes': byte_count,
            'tokens': token_count,
            'ratio': ratio
        })

        total_bytes += byte_count
        total_tokens += token_count

        print(f"  Doc {i+1}: {byte_count:,} bytes, {token_count:,} tokens, {ratio:.2f} bytes/token")

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    overall_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    throughput = total_bytes / elapsed_time if elapsed_time > 0 else 0

    return {
        'total_bytes': total_bytes,
        'total_tokens': total_tokens,
        'compression_ratio': overall_ratio,
        'doc_stats': doc_stats,
        'elapsed_time': elapsed_time,
        'throughput': throughput
    }


def estimate_pile_time(throughput_bytes_per_sec: float, pile_size_gb: float = 825) -> dict:
    """
    Estimate time to tokenize the Pile dataset.
    
    Args:
        throughput_bytes_per_sec: Throughput in bytes/second
        pile_size_gb: Size of Pile dataset in GB (default: 825)
    
    Returns:
        Dictionary with time estimates
    """
    pile_bytes = pile_size_gb * 1024 ** 3  # Convert GB to bytes
    time_seconds = pile_bytes / throughput_bytes_per_sec
    
    time_minutes = time_seconds / 60
    time_hours = time_minutes / 60
    time_days = time_hours / 24
    
    return {
        'pile_size_gb': pile_size_gb,
        'pile_bytes': pile_bytes,
        'time_seconds': time_seconds,
        'time_minutes': time_minutes,
        'time_hours': time_hours,
        'time_days': time_days
    }


def main():
    # Paths to datasets
    tinystories_path = "/ocean/projects/cis250265p/xli45/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    owt_path = "/ocean/projects/cis250265p/xli45/assignment1-basics/data/owt_train.txt"

    # Paths to trained tokenizers
    tinystories_vocab = "/ocean/projects/cis250265p/xli45/assignment1-basics/trained_tokenizers/tinystories/vocab.json"
    tinystories_merges = "/ocean/projects/cis250265p/xli45/assignment1-basics/trained_tokenizers/tinystories/merges.txt"

    owt_vocab = "/ocean/projects/cis250265p/xli45/assignment1-basics/trained_tokenizers/owt/vocab.json"
    owt_merges = "/ocean/projects/cis250265p/xli45/assignment1-basics/trained_tokenizers/owt/merges.txt"

    num_samples = 10

    print("=" * 80)
    print("Compression Ratio Experiment")
    print("=" * 80)

    # Load TinyStories tokenizer (10K vocab)
    print("\n--- TinyStories Tokenizer (10K vocab) ---")
    ts_tokenizer = BPETokenizer.from_files(
        tinystories_vocab,
        tinystories_merges,
        special_tokens=["<|endoftext|>"]
    )
    print(f"Vocabulary size: {len(ts_tokenizer.vocab)}")

    # Sample TinyStories documents
    print(f"\nSampling {num_samples} documents from TinyStories...")
    ts_documents = sample_documents(tinystories_path, num_samples)

    # Calculate compression ratio for TinyStories
    print("\nCalculating compression ratio for TinyStories:")
    ts_results = calculate_compression_ratio(ts_tokenizer, ts_documents)

    print(f"\nTinyStories Summary:")
    print(f"  Total bytes: {ts_results['total_bytes']:,}")
    print(f"  Total tokens: {ts_results['total_tokens']:,}")
    print(f"  Compression ratio: {ts_results['compression_ratio']:.4f} bytes/token")
    print(f"  Processing time: {ts_results['elapsed_time']:.4f} seconds")
    print(f"  Throughput: {ts_results['throughput']:,.2f} bytes/second ({ts_results['throughput']/1024**2:.2f} MB/s)")

    # Load OpenWebText tokenizer (32K vocab)
    print("\n" + "=" * 80)
    print("\n--- OpenWebText Tokenizer (32K vocab) ---")
    owt_tokenizer = BPETokenizer.from_files(
        owt_vocab,
        owt_merges,
        special_tokens=["<|endoftext|>"]
    )
    print(f"Vocabulary size: {len(owt_tokenizer.vocab)}")

    # Sample OpenWebText documents
    print(f"\nSampling {num_samples} documents from OpenWebText...")
    owt_documents = sample_documents(owt_path, num_samples)

    # Calculate compression ratio for OpenWebText
    print("\nCalculating compression ratio for OpenWebText:")
    owt_results = calculate_compression_ratio(owt_tokenizer, owt_documents)

    print(f"\nOpenWebText Summary:")
    print(f"  Total bytes: {owt_results['total_bytes']:,}")
    print(f"  Total tokens: {owt_results['total_tokens']:,}")
    print(f"  Compression ratio: {owt_results['compression_ratio']:.4f} bytes/token")
    print(f"  Processing time: {owt_results['elapsed_time']:.4f} seconds")
    print(f"  Throughput: {owt_results['throughput']:,.2f} bytes/second ({owt_results['throughput']/1024**2:.2f} MB/s)")

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"TinyStories (10K vocab): {ts_results['compression_ratio']:.4f} bytes/token")
    print(f"OpenWebText (32K vocab): {owt_results['compression_ratio']:.4f} bytes/token")
    
    # Estimate Pile tokenization time
    print("\n" + "=" * 80)
    print("PILE DATASET TOKENIZATION ESTIMATE (825 GB)")
    print("=" * 80)
    
    # Use average throughput
    avg_throughput = (ts_results['throughput'] + owt_results['throughput']) / 2
    pile_estimate = estimate_pile_time(avg_throughput)
    
    print(f"Average throughput: {avg_throughput:,.2f} bytes/second ({avg_throughput/1024**2:.2f} MB/s)")
    print(f"\nEstimated time to tokenize 825 GB Pile dataset:")
    print(f"  {pile_estimate['time_seconds']:,.0f} seconds")
    print(f"  {pile_estimate['time_minutes']:,.0f} minutes")
    print(f"  {pile_estimate['time_hours']:,.2f} hours")
    print(f"  {pile_estimate['time_days']:.2f} days")
    
    print("\n" + "=" * 80)

    return ts_results, owt_results, pile_estimate


if __name__ == "__main__":
    main()