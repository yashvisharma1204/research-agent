"""
graph/merger.py
Responsible for upserting entities and triples into Neo4j.
Key feature: fuzzy entity resolution prevents duplicate nodes
when the same real-world entity appears under slightly different names.
"""
from __future__ import annotations

import logging

import numpy as np
from neo4j import Driver
from sentence_transformers import SentenceTransformer

from config import cfg

logger = logging.getLogger(__name__)

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model
