"""
Lightweight embedder using ONNX — no torch required.
Uses the same all-MiniLM-L6-v2 model via HuggingFace tokenizers + onnxruntime.
"""
from __future__ import annotations
import numpy as np

_session = None
_tokenizer = None

def _load():
    global _session, _tokenizer
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
