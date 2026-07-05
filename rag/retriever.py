"""
rag/retriever.py
Two retrieval strategies, fused into a single context bundle:
  1. Graph traversal — multi-hop Cypher over Neo4j
  2. Vector search   — Native Neo4j Vector index similarity

The QueryRouter (router.py) decides which strategy/strategies to use.
"""
from __future__ import annotations

import logging
from typing import NamedTuple
from neo4j import Driver

from config import cfg
from rag.router import route_query
from utils.embedder import encode as _encode

logger = logging.getLogger(__name__)

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
    route: str = "hybrid"


# ── Unified Graph & Vector Retriever ──────────────────────────────────────────

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
        limit: int | None = None,
    ) -> list[GraphTriple]:
        limit = limit or cfg.GRAPH_TRIPLE_LIMIT
        if not entity_names:
            return []
            
        cypher = """
        MATCH (seed:Entity)
        WHERE seed.name IN $seeds
        OR ANY(s IN $seeds WHERE toLower(seed.name) CONTAINS toLower(s))
        MATCH (seed)-[r]-(neighbor:Entity)
        RETURN seed.name AS subject, r.type AS predicate,
            neighbor.name AS obj,
            coalesce(r.confidence, 1.0) AS confidence,
            coalesce(r.mention_count, 1) AS mention_count
        ORDER BY r.confidence DESC
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

    def vector_search(self, query: str, k: int = 5) -> list[VectorResult]:
        """Queries the Native Neo4j vector index."""
        q_emb = _encode([query])[0].tolist()
        cypher = """
        CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding)
        YIELD node, score
        RETURN node.text AS text, score, node.name AS doc_id
        """
        try:
            with self.driver.session() as s:
                records = s.run(cypher, k=k, embedding=q_emb).data()
            return [
                VectorResult(text=r["text"], score=r["score"], doc_id=r["doc_id"])
                for r in records if r.get("text")
            ]
        except Exception as exc:
            logger.warning("Neo4j Vector search failed (is the index built?): %s", exc)
            return []


# ── Retriever orchestrator ────────────────────────────────────────────────────

class Retriever:
    def __init__(self, driver: Driver):
        self.driver = driver
        self.graph = GraphRetriever(driver)

    def retrieve(self, query: str) -> RetrievalContext:
        route = route_query(query)
        entity_names = self.graph.extract_entities_from_query(query)
        
        if not entity_names and route in ("cypher", "hybrid"):
            entity_names = self.graph.fulltext_entity_search(query)

        graph_triples: list[GraphTriple] = []
        if route in ("cypher", "hybrid"):
            graph_triples = self.graph.multi_hop_traverse(entity_names)
        
        if route == "cypher" and len(graph_triples) < 3:
            logger.info("Cypher returned %d triples — falling back to vector", len(graph_triples))
            route = "hybrid"

        vector_results: list[VectorResult] = []
        if route in ("vector", "hybrid"):
            vector_results = self.graph.vector_search(query, k=cfg.VECTOR_TOP_K)

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
        """Generates an embedding and saves the text/vector directly to the Neo4j Paper node."""
        text = f"{doc.get('title', '')}. {doc.get('abstract', '')}"
        emb = _encode([text])[0].tolist()
        doc_id = doc.get("title", doc.get("id", ""))
        
        cypher = """
        MATCH (e:Entity {name: $name})
        SET e.text = $text, e.embedding = $emb
        """
        with self.driver.session() as s:
            s.run(cypher, name=doc_id, text=text, emb=emb)
        logger.info("Vector embedding saved to Neo4j for node: %s", doc_id)