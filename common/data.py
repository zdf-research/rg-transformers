import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def download_file(url: str, filename: str) -> str:
    """Downloads a file if it doesn't already exist and returns the absolute path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.abspath(os.path.join(DATA_DIR, filename))
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")
    return filepath

def get_tinyshakespeare() -> str:
    """
    Returns the path to the TinyShakespeare dataset (~1MB).
    Excellent for quick debugging runs, overfitting tests, and character-level models.
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    return download_file(url, "tinyshakespeare.txt")

def get_wikitext() -> str:
    """
    Returns the path to WikiText-2 train split (~14MB).
    Clean Wikipedia text that is highly representative of standard language modelling datasets.
    Perfect size for training a local BPE tokenizer in a few minutes on CPU.
    """
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    return download_file(url, "wikitext.txt")
