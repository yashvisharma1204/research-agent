"""
rag/retriever.py
Two retrieval strategies, fused into a single context bundle:
  1. Graph traversal — multi-hop Cypher over Neo4j
  2. Vector search    — FAISS similarity over paper abstracts

The QueryRouter (router.py) decides which strategy/strategies to use.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import NamedTuple

import faiss
import numpy as np
from neo4j import Driver

from config import cfg
from rag.router import route_query

logger = logging.getLogger(__name__)

_FAISS_INDEX_PATH = Path("data/faiss.index")
_FAISS_META_PATH = Path("data/faiss_meta.pkl")
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_embed_model = None


def _model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model

# ── Data types ─────────────────────────────────────────────────────────────────

class GraphTriple(NamedTuple):
    subject: str
    predicate: str
    obj: str
    confidence: float
    mention_count: int


class VectorResult(NamedTuple):
    text: str
    score: float
    doc_id: str


class RetrievalContext(NamedTuple):
    graph_triples: list[GraphTriple]
    vector_results: list[VectorResult]
    entity_names: list[str]
    route: str = "hybrid"   # which route was used


# ── FAISS vector store ────────────────────────────────────────────────────────

class VectorStore:
    """Lightweight FAISS store that persists to disk."""

    def __init__(self):
        self._index: faiss.IndexFlatIP | None = None
        self._meta: list[dict] = []
        self._load()

    def _load(self):
        if _FAISS_INDEX_PATH.exists() and _FAISS_META_PATH.exists():
            self._index = faiss.read_index(str(_FAISS_INDEX_PATH))
            with open(_FAISS_META_PATH, "rb") as f:
                self._meta = pickle.load(f)
            logger.info("FAISS index loaded (%d vectors)", self._index.ntotal)
        else:
            dim = 384   # all-MiniLM-L6-v2 output dim
            self._index = faiss.IndexFlatIP(dim)
            self._meta = []

    def _save(self):
        _FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(_FAISS_INDEX_PATH))
        with open(_FAISS_META_PATH, "wb") as f:
            pickle.dump(self._meta, f)

    def add(self, texts: list[str], doc_ids: list[str]):
        if not texts:
            return
        embs = _model().encode(texts, normalize_embeddings=True).astype("float32")
        self._index.add(embs)
        self._meta.extend({"text": t, "doc_id": d} for t, d in zip(texts, doc_ids))
        self._save()
        logger.debug("Added %d vectors to FAISS", len(texts))

    def search(self, query: str, k: int = 5) -> list[VectorResult]:
        if self._index.ntotal == 0:
            return []
        q_emb = _model().encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(q_emb, min(k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._meta[idx]
            results.append(VectorResult(
                text=meta["text"],
                score=float(score),
                doc_id=meta["doc_id"],
            ))
        return results


# ── Graph traversal ───────────────────────────────────────────────────────────

class GraphRetriever:
    def __init__(self, driver: Driver):
        self.driver = driver

    def extract_entities_from_query(self, query: str) -> list[str]:
        import re
        quoted = re.findall(r'"([^"]+)"', query)
        capitalized = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b', query)
        single_caps = re.findall(r'\b([A-Z]{2,})\b', query)
        return list(dict.fromkeys(quoted + capitalized + single_caps))

    def multi_hop_traverse(
        self,
        entity_names: list[str],
        hops: int | None = None,
        limit: int | None = None,
    ) -> list[GraphTriple]:
        hops = hops or cfg.GRAPH_HOP_LIMIT
        limit = limit or cfg.GRAPH_TRIPLE_LIMIT

        if not entity_names:
            return []

        cypher = f"""
        MATCH path = (seed:Entity)-[*1..{hops}]-(neighbor:Entity)
        WHERE seed.name IN $seeds
        WITH relationships(path) AS rels
        UNWIND rels AS r
        WITH startNode(r) AS s, r, endNode(r) AS o
        RETURN
          s.name  AS subject,
          r.type  AS predicate,
          o.name  AS obj,
          coalesce(r.confidence, 1.0)    AS confidence,
          coalesce(r.mention_count, 1)   AS mention_count
        ORDER BY r.confidence DESC, r.mention_count DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            records = session.run(cypher, seeds=entity_names, limit=limit).data()

        return [
            GraphTriple(
                subject=r["subject"],
                predicate=r["predicate"],
                obj=r["obj"],
                confidence=r["confidence"],
                mention_count=r["mention_count"],
            )
            for r in records
        ]

    def fulltext_entity_search(self, query: str, limit: int = 10) -> list[str]:
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
        YIELD node, score
        RETURN node.name AS name
        ORDER BY score DESC
        LIMIT $limit
        """
        try:
            with self.driver.session() as s:
                return [r["name"] for r in s.run(cypher, query=query, limit=limit).data()]
        except Exception as exc:
            logger.warning("Fulltext search failed: %s", exc)
            return []


# ── Unified retriever ─────────────────────────────────────────────────────────

class Retriever:
    def __init__(self, driver: Driver):
        self.graph = GraphRetriever(driver)
        self.vector = VectorStore()

    def retrieve(self, query: str) -> RetrievalContext:
        # 1. Classify intent
        route = route_query(query)

        # 2. Find seed entities from query text
        entity_names = self.graph.extract_entities_from_query(query)
        if not entity_names and route in ("cypher", "hybrid"):
            entity_names = self.graph.fulltext_entity_search(query)

        # 3. Graph traversal (skip for pure vector route)
        graph_triples: list[GraphTriple] = []
        if route in ("cypher", "hybrid"):
            graph_triples = self.graph.multi_hop_traverse(entity_names)

        # 4. Vector search (skip for pure cypher route)
        vector_results: list[VectorResult] = []
        if route in ("vector", "hybrid"):
            vector_results = self.vector.search(query, k=cfg.VECTOR_TOP_K)

        logger.info(
            "Retrieval [%s]: %d entities, %d graph triples, %d vector results",
            route, len(entity_names), len(graph_triples), len(vector_results),
        )
        return RetrievalContext(
            graph_triples=graph_triples,
            vector_results=vector_results,
            entity_names=entity_names,
            route=route,
        )

    def index_document(self, doc: dict):
        """Add a document's abstract to the FAISS vector index."""
        text = f"{doc.get('title', '')}. {doc.get('abstract', '')}"
        self.vector.add([text], [doc.get("id", "")])
