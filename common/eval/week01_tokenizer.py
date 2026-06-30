import os
import time
import pytest
import importlib

# To run manually: 
# MEMBER_NAME=alice pytest common/eval/week01_tokenizer.py --junitxml=results.xml

@pytest.fixture(scope="module")
def member_tokenizer():
    """
    Dynamically loads the tokenizer implementation for the specified member.
    The GitHub Action will inject the MEMBER_NAME environment variable.
    """
    member_name = os.environ.get("MEMBER_NAME")
    if not member_name:
        pytest.fail("MEMBER_NAME environment variable is required to run the evaluation.")
        
    module_path = f"implementations.{member_name}.week01_embeddings"
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        pytest.fail(f"Could not find {module_path}. Make sure the file exists.")
        
    if not hasattr(module, "BPETokenizer"):
        pytest.fail("The module must expose a class named 'BPETokenizer'.")
        
    # Instantiate the member's tokenizer
    return module.BPETokenizer()

@pytest.fixture(scope="module")
def text_corpus():
    """Loads a slice of text to test parity and compression."""
    # In practice, we load WikiText-2 here using common.data
    return "The quick brown fox jumps over the lazy dog. " * 100

def test_encode_decode_parity(member_tokenizer, text_corpus):
    """Metric: Encode/Decode Parity"""
    # Assuming the tokenizer is either trained or we train it here
    ids = member_tokenizer.encode(text_corpus)
    decoded_text = member_tokenizer.decode(ids)
    
    assert text_corpus == decoded_text, "Decode(Encode(text)) does not match the original text!"

def test_compression_ratio_and_speed(member_tokenizer, text_corpus, record_property):
    """Metric: Compression Ratio and Lookup Speed"""
    raw_bytes = len(text_corpus.encode("utf-8"))
    
    start_time = time.time()
    ids = member_tokenizer.encode(text_corpus)
    latency_ms = (time.time() - start_time) * 1000
    
    # Calculate bytes per token
    compression_ratio = raw_bytes / len(ids) if len(ids) > 0 else 0.0
    
    # Record properties so the GitHub Action can parse them into the leaderboard
    record_property("compression_ratio", round(compression_ratio, 2))
    record_property("latency_ms", round(latency_ms, 2))
    
    assert compression_ratio > 1.0, f"Failed: Compression ratio ({compression_ratio}) is worse than character-level."
