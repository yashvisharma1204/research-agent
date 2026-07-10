"""
rag/retriever.py
Two retrieval strategies, fused into a single context bundle:
  1. Graph traversal — Multi-hop Cypher reasoning over Neo4j (Day 10)
  2. Vector search   — Native Neo4j Vector index similarity
"""
from __future__ import annotations

import logging
import math
from typing import NamedTuple
from neo4j import Driver
import google.generativeai as genai

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
    retrieval_score: float = 0.0


class VectorResult(NamedTuple):
    text: str
    score: float
    doc_id: str


class RetrievalContext(NamedTuple):
    graph_triples: list[GraphTriple]
    vector_results: list[VectorResult]
    entity_names: list[str]
    paper_abstracts: list[str] = [] 
    route: str = "hybrid"


# ── Unified Graph & Vector Retriever ──────────────────────────────────────────

class GraphRetriever:
    def __init__(self, driver: Driver):
        self.driver = driver
        genai.configure(api_key=cfg.GEMINI_API_KEY)
        self._extractor_model = genai.GenerativeModel(
            model_name=cfg.LLM_MODEL,
            system_instruction="You are an entity extractor for a knowledge graph. Extract the core technical concepts, AI models, or frameworks from the query. Return ONLY a comma-separated list of terms. Do not include markdown, explanations, or quotes."
        )

    def extract_entities_from_query(self, query: str) -> list[str]:
        """Uses Gemini to intelligently extract entities instead of regex."""
        try:
            response = self._extractor_model.generate_content(query)
            if response.text:
                entities = [e.strip() for e in response.text.split(",") if e.strip()]
                logger.debug("Gemini extracted entities: %s", entities)
                return entities
            return []
        except Exception as exc:
            logger.warning("Gemini entity extraction failed: %s", exc)
            return []

    def multi_hop_traverse(
        self,
        entity_names: list[str],
        limit: int | None = None,
    ) -> list[GraphTriple]:
        """
        Day 10: Traverses up to 2 hops across the graph to extract contextual paths.
        Day 9: Employs structural density math to calculate accurate relevance scores.
        """
        limit = limit or cfg.GRAPH_TRIPLE_LIMIT
        if not entity_names:
            return []
            
        # Day 10: Variable length multi-hop matching (1 to 2 hops deep)
        # Avoids tracking circular paths back to the starting seed node.
        cypher = """
        MATCH (seed:Entity)
        WHERE ANY(s IN $seeds WHERE toLower(seed.name) CONTAINS toLower(s))
        MATCH path = (seed)-[r*1..2]-(neighbor:Entity)
        WHERE seed <> neighbor
        UNWIND relationships(path) AS rel
        WITH startNode(rel) AS s_node, endNode(rel) AS t_node, rel
        RETURN DISTINCT 
            s_node.name AS subject, 
            type(rel) AS predicate,
            t_node.name AS obj,
            coalesce(rel.confidence, 1.0) AS confidence,
            coalesce(rel.mention_count, 1) AS mention_count
        """
        
        with self.driver.session() as session:
            records = session.run(cypher, seeds=entity_names).data()

        processed_triples = []
        for r in records:
            conf = float(r["confidence"])
            mentions = int(r["mention_count"])
            
            # Day 9 Optimization: Confidence combined with logarithmic structural scale
            score = conf * math.log(mentions + 1)
            
            processed_triples.append(
                GraphTriple(
                    subject=r["subject"],
                    predicate=r["predicate"],
                    obj=r["obj"],
                    confidence=conf,
                    mention_count=mentions,
                    retrieval_score=score
                )
            )

        # Rank all compiled multi-hop paths based on their structural scores
        processed_triples.sort(key=lambda x: x.retrieval_score, reverse=True)
        return processed_triples[:limit]
    
    def get_abstracts_for_entities(self, entity_names: list[str]) -> list[str]:
        """Day 11: Fetches raw text/abstracts for nodes involved in the multi-hop paths."""
        if not entity_names:
            return []
            
        cypher = """
        MATCH (n:Entity)
        // Match nodes by name and ensure they actually contain text data
        WHERE n.name IN $names AND n.text IS NOT NULL
        RETURN n.name AS name, n.text AS text
        """
        
        try:
            with self.driver.session() as session:
                records = session.run(cypher, names=entity_names).data()
            # Format the output so Gemini knows exactly which paper the text belongs to
            return [f"[{r['name']} Abstract]: {r['text']}" for r in records]
        except Exception as exc:
            logger.warning("Failed to fetch paper abstracts: %s", exc)
            return []

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
        
        # 1. Gemini Entity Extraction
        entity_names = self.graph.extract_entities_from_query(query)
        
        # 2. Fallback to Fulltext if Gemini fails to find anything
        if not entity_names and route in ("cypher", "hybrid"):
            entity_names = self.graph.fulltext_entity_search(query)

        graph_triples: list[GraphTriple] = []
        if route in ("cypher", "hybrid"):
            graph_triples = self.graph.multi_hop_traverse(entity_names)
        
        # 3. Cypher Fallback Logic (Automatically forces hybrid/vector if graph is empty)
        if route == "cypher" and len(graph_triples) == 0:
            logger.info("Cypher returned 0 triples — auto-falling back to hybrid vector search")
            route = "hybrid"

        vector_results: list[VectorResult] = []
        if route in ("vector", "hybrid"):
            vector_results = self.graph.vector_search(query, k=cfg.VECTOR_TOP_K)

        paper_abstracts = []
        if graph_triples:
            # Extract a unique set of every node name present in our top paths
            unique_nodes = {t.subject for t in graph_triples}.union({t.obj for t in graph_triples})
            paper_abstracts = self.graph.get_abstracts_for_entities(list(unique_nodes))

        logger.info(
            "Retrieval [%s]: %d entities, %d triples, %d abstracts, %d vectors",
            route, len(entity_names), len(graph_triples), len(paper_abstracts), len(vector_results)
        )
        
        return RetrievalContext(
            graph_triples=graph_triples,
            vector_results=vector_results,
            entity_names=entity_names,
            paper_abstracts=paper_abstracts, 
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