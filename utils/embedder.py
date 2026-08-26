"""
Lightweight embedder using ONNX — no torch required.
Uses the same all-MiniLM-L6-v2 model via HuggingFace tokenizers + onnxruntime.
"""
from __future__ import annotations
import threading
import numpy as np

_session = None
_tokenizer = None
_lock = threading.Lock()

def _load():
    global _session, _tokenizer
    if _session is not None:
        return
    with _lock:
        if _session is None:
            from transformers import AutoTokenizer
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            model_id = "sentence-transformers/all-MiniLM-L6-v2"
            _tokenizer = AutoTokenizer.from_pretrained(model_id)
            onnx_path = hf_hub_download(model_id, filename="onnx/model.onnx")
            _session = ort.InferenceSession(onnx_path)

def encode(texts: list[str], normalize: bool = True) -> np.ndarray:
    _load()
    inputs = _tokenizer(
        texts, padding=True, truncation=True,
        max_length=256, return_tensors="np"
    )
    outputs = _session.run(None, dict(inputs))
    # Mean pool token embeddings
    embeddings = outputs[0].mean(axis=1)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)
    return embeddings.astype("float32")


def chunk_text(
    text: str,
    chunk_words: int = 120,
    overlap_words: int = 30,
) -> list[str]:
    """Split *text* into overlapping word-windows suitable for passage embedding.

    Uses whole-word windows (not tokens) so chunks stay human-readable and align
    with the embedder's 256-token truncation limit — ~120 words comfortably fits
    under 256 tokens for English prose. Overlap preserves context across chunk
    boundaries so a relationship split across two sentences is still retrievable.

    Returns a list with at least one chunk for any non-empty input.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_words:
        return [" ".join(words)]

    step = max(1, chunk_words - overlap_words)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks
