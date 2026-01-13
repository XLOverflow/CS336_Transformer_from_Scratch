from typing import List, Tuple, Dict, Iterable, Iterator
import json
import base64
import regex as re

# GPT-2 regex pattern for pre-tokenization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
        if self.special_tokens is None or len(self.special_tokens) == 0:
            return [(text, False)]

        # Sort special tokens by length (longest first) to prioritize longer matches
        # This handles overlapping special tokens correctly
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

        # Create a regex pattern to match special tokens
        special_pattern = '|'.join(re.escape(token) for token in sorted_special_tokens)

        spilt_text = re.split(f'({special_pattern})', text)

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
                matchs = re.findall(PAT, part)

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


    