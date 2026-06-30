from typing import List, Protocol

class Tokenizer(Protocol):
    """
    Standard interface for all reading group tokenizers.
    Whether building a character-level tokenizer or a BPE tokenizer,
    every implementation must conform to this contract to ensure compatibility
    with the shared evaluation scripts.
    """
    
    @property
    def vocab_size(self) -> int:
        """Returns the total number of unique tokens in the vocabulary."""
        ...
        
    def train(self, text: str, vocab_size: int, **kwargs) -> None:
        """
        Trains the tokenizer on the provided text.
        For character-level, this identifies unique characters.
        For BPE, this iteratively merges the most frequent pairs until vocab_size is reached.
        """
        ...
        
    def encode(self, text: str) -> List[int]:
        """Converts a string of text into a sequence of token IDs."""
        ...
        
    def decode(self, ids: List[int]) -> str:
        """Converts a sequence of token IDs back into a string of text."""
        ...
        
    def save(self, filepath: str) -> None:
        """Saves the vocabulary (and merges for BPE) to disk."""
        ...
        
    def load(self, filepath: str) -> None:
        """Loads the vocabulary (and merges) from disk."""
        ...
