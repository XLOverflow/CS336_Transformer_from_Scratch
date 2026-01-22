from typing import List, Tuple, Dict, Iterable, Iterator
import json
import base64
import os
import regex as re
from multiprocessing import Pool, cpu_count

# GPT-2 regex pattern for pre-tokenization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Global tokenizer and chunks for worker processes (initialized once per worker)
_worker_tokenizer = None
_worker_chunks = None

def _init_worker(vocab, merges, special_tokens, chunks):
    """Initialize tokenizer and chunks in worker process."""
    global _worker_tokenizer, _worker_chunks
    _worker_tokenizer = BPETokenizer(vocab, merges, special_tokens)
    _worker_chunks = chunks

def _encode_batch_and_save(args):
    """Encode a batch of chunks and save to tmp file."""
    batch_id, start_idx, end_idx, tmp_dir = args
    import numpy as np

    results = []
    for i in range(start_idx, end_idx):
        idx, chunk = _worker_chunks[i]
        ids = _worker_tokenizer.encode(chunk)
        results.append((idx, ids))

    # Save to tmp file
    np.save(os.path.join(tmp_dir, f"batch_{batch_id}.npy"),
            np.array(results, dtype=object), allow_pickle=True)
    return batch_id

class BPETokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str]=None):
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens.copy()

        # Add special tokens to the vocabulary if not already present
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in self.vocab.values():
                self.vocab[max(self.vocab.keys()) + 1] = token_bytes

        # Build reverse vocab mapping: bytes -> id
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}

        # Build merge ranks for efficient BPE application
        # Lower rank = higher priority (earlier in merge list)
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        
        # Pre-compile regex patterns for performance
        self.pat_compiled = re.compile(PAT)
        
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = '|'.join(re.escape(token) for token in sorted_special_tokens)
            self.special_pattern_compiled = re.compile(f'({special_pattern})')
        else:
            self.special_pattern_compiled = None

    @staticmethod
    def _load_vocab(vocab_filepath: str) -> Dict[int, bytes]:
        """
        Load vocabulary from a JSON file.

        JSON format: {base64_encoded_token: token_id}
        Returns: {token_id: token_bytes}
        """
        with open(vocab_filepath, "r", encoding='utf-8') as f:
            vocab_str2id = json.load(f)

        vocab = {}
        for token_str, token_id in vocab_str2id.items():
            # Decode base64 string back to bytes
            token_bytes = base64.b64decode(token_str.encode('ascii'))
            vocab[token_id] = token_bytes

        return vocab

    @staticmethod
    def _load_merges(merges_filepath: str) -> List[Tuple[bytes, bytes]]:
        """
        Load merges from a text file.

        Format: Each line contains two base64-encoded tokens separated by space
        Returns: List of (token1_bytes, token2_bytes) tuples
        """
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:  # Skip empty lines
                    continue
                # Split by space to get two base64-encoded tokens
                parts = line.split(' ')
                if len(parts) == 2:
                    token1_bytes = base64.b64decode(parts[0].encode('ascii'))
                    token2_bytes = base64.b64decode(parts[1].encode('ascii'))
                    merges.append((token1_bytes, token2_bytes))

        return merges

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str]=None):
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)
    
    def _split_by_special_tokens(self, text: str) -> List[Tuple[str, bool]]:
        """
        Split text into segments based on special tokens.

        Returns a list of tuples (segment, is_special_token)
        where is_special_token is True if the segment is a special token.
        """
        if not self.special_pattern_compiled:
            return [(text, False)]

        spilt_text = self.special_pattern_compiled.split(text)

        result = []
        for part in spilt_text:
            if not part:
                continue
            result.append((part, part in self.special_tokens))

        return result
    
    def _apply_merge(self, token_bytes: bytes) -> List[bytes]:
        """
        Apply BPE merges to a single token using greedy algorithm.

        Algorithm:
        1. Start with individual bytes
        2. Repeatedly find the pair with lowest rank (highest priority)
        3. Merge that pair and repeat until no more merges available

        Time complexity: O(n^2) where n is token length, much better than O(merges × n)

        Args:
            token_bytes: Input token as bytes

        Returns:
            List of byte tokens after applying BPE merges
        """
        # Edge case: empty or single byte
        if len(token_bytes) <= 1:
            return [token_bytes] if token_bytes else []

        # Initialize with individual bytes
        tokens = [bytes([b]) for b in token_bytes]

        # Greedy algorithm: repeatedly merge the highest priority pair
        while len(tokens) > 1:
            # Find all adjacent pairs and their ranks
            min_rank = float('inf')
            min_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair, float('inf'))

                if rank < min_rank:
                    min_rank = rank
                    min_idx = i

            # If no valid merge found, we're done
            if min_rank == float('inf'):
                break

            # Merge the best pair
            tokens[min_idx] = tokens[min_idx] + tokens[min_idx + 1]
            del tokens[min_idx + 1]

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode a single string into a list of token IDs.
        
        Steps:
        1. Pre-tokenize the input text using the GPT-2 regex pattern(keep special tokens).
        2. Apply BPE merges to each token.
        3. Convert the resulting byte tokens to their corresponding IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        ids = []

        # Step 1: Split text by special tokens
        text_parts = self._split_by_special_tokens(text)

        for part, is_special in text_parts:
            if is_special:
                # we can directly map special_tokens to its ID
                special_token_bytes = part.encode('utf-8')
                if special_token_bytes in self.vocab_reversed:
                    ids.append(self.vocab_reversed[special_token_bytes])
                else:
                    raise ValueError(f"Special token '{part}' not found in vocabulary")
            else:
                # for regular text, we need to pre_tokenization and apply BPE, then converting to IDs
                matchs = self.pat_compiled.findall(part)

                for match in matchs:
                    matchs_bytes = match.encode('utf-8')
                    # Apply BPE merges
                    bpe_tokens = self._apply_merge(matchs_bytes)
                    # Convert byte tokens to IDs
                    for token in bpe_tokens:
                        if token in self.vocab_reversed:
                            ids.append(self.vocab_reversed[token])
                        else:
                            raise ValueError(f"Token '{token}' not found in vocabulary")
        
        return ids

    def encode_parallel(self, text: str, num_processes: int = None,
                       split_token: str = "<|endoftext|>",
                       batch_size: int = 100000,
                       output_file: str = None) -> List[int]:
        """
        Encode text in parallel by splitting on special tokens.

        Uses batch + checkpoint approach: each process handles a batch of chunks,
        saves result to tmp file, then moves to next batch. Supports resume.

        Args:
            text: Input text to encode
            num_processes: Number of processes to use (default: cpu_count())
            split_token: Special token to split on for parallelization
            batch_size: Number of chunks per batch (default: 100000)
            output_file: Output file path for encoded data

        Returns:
            memmap array of token IDs
        """
        from tqdm import tqdm
        import numpy as np

        if num_processes is None:
            num_processes = cpu_count()

        if output_file is None:
            output_file = "./encoded.npy"

        # Use output file name (without extension) as tmp directory
        base_name = os.path.splitext(output_file)[0]
        tmp_dir = base_name + "_tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        # Split text by the special token
        if split_token not in text:
            return self.encode(text)

        chunks = text.split(split_token)
        chunks_with_idx = []
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                chunks_with_idx.append((i, chunk + split_token))
            elif chunk:  # Last chunk only if non-empty
                chunks_with_idx.append((i, chunk))

        if not chunks_with_idx:
            return []

        num_chunks = len(chunks_with_idx)
        num_batches = (num_chunks + batch_size - 1) // batch_size

        print(f"Total: {num_chunks:,} chunks, {num_batches} batches (batch_size={batch_size})")

        # Check which batches are already done
        completed = set()
        for batch_id in range(num_batches):
            if os.path.exists(os.path.join(tmp_dir, f"batch_{batch_id}.npy")):
                completed.add(batch_id)

        # Build pending batches (only pass indices, not data)
        pending_batches = []
        for batch_id in range(num_batches):
            if batch_id not in completed:
                start = batch_id * batch_size
                end = min(start + batch_size, num_chunks)
                pending_batches.append((batch_id, start, end, tmp_dir))

        print(f"Completed: {len(completed)}, Pending: {len(pending_batches)}")

        # Process pending batches
        if pending_batches:
            with Pool(processes=num_processes,
                      initializer=_init_worker,
                      initargs=(self.vocab, self.merges, self.special_tokens, chunks_with_idx)) as pool:
                with tqdm(total=len(pending_batches), desc="Encoding batches") as pbar:
                    for batch_id in pool.imap_unordered(_encode_batch_and_save, pending_batches):
                        pbar.update(1)

        # Merge results and save to final file (stream to avoid OOM)
        if not os.path.exists(output_file):
            print("Merging results to final file...")

            # First pass: count total tokens per batch
            batch_tokens = []
            for batch_id in tqdm(range(num_batches), desc="Counting"):
                batch_file = os.path.join(tmp_dir, f"batch_{batch_id}.npy")
                batch_data = np.load(batch_file, allow_pickle=True)
                count = sum(len(ids) for idx, ids in batch_data)
                batch_tokens.append(count)

            total_tokens = sum(batch_tokens)
            print(f"Total tokens: {total_tokens:,}")

            # Create memmap file
            final_arr = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(total_tokens,))

            # Second pass: write batches in order (batch内按idx排序)
            print("Writing to final file...")
            offset = 0
            for batch_id in tqdm(range(num_batches), desc="Writing"):
                batch_file = os.path.join(tmp_dir, f"batch_{batch_id}.npy")
                batch_data = np.load(batch_file, allow_pickle=True)
                # Sort by idx within batch to ensure correct order
                batch_sorted = sorted(batch_data, key=lambda x: x[0])
                for idx, ids in batch_sorted:
                    final_arr[offset:offset + len(ids)] = ids
                    offset += len(ids)

            final_arr.flush()
            del final_arr
            print(f"Saved to {output_file}")

        # Return memmap array (doesn't load into memory)
        print(f"Loading from {output_file}...")
        return np.memmap(output_file, dtype=np.uint16, mode='r')
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of text chunks, yielding token IDs.

        This method handles streaming/chunked encoding efficiently.

        IMPORTANT: To ensure correct tokenization across chunks:
        - Chunks should be split at natural boundaries (e.g., after complete sentences)
        - Ideally, split at special token boundaries to avoid cross-chunk tokens
        - Each chunk is encoded independently

        Args:
            iterable: An iterable of text strings

        Yields:
            Individual token IDs
        """
        for text in iterable:
            ids = self.encode(text)
            for token_id in ids:
                yield token_id

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a string.

        Steps:
        1. Map each token ID to its corresponding bytes using vocab
        2. Concatenate all bytes together
        3. Decode the byte sequence as UTF-8

        Args:
            ids: List of token IDs to decode

        Returns:
            Decoded string

        Raises:
            ValueError: If any token ID is not found in the vocabulary
            UnicodeDecodeError: If the byte sequence is not valid UTF-8
        """
        # Step 1 & 2: Map IDs to bytes and concatenate
        byte_sequence = b''
        for token_id in ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocabulary")
            byte_sequence += self.vocab[token_id]

        # Step 3: Decode bytes to string
        # Use 'replace' error handling to avoid crashes on invalid UTF-8
        # (though with proper BPE, this shouldn't happen)
        return byte_sequence.decode('utf-8', errors='replace')