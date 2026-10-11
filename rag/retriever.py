"""
rag/retriever.py
Two retrieval strategies, fused into a single context bundle:
  1. Graph traversal — Multi-hop Cypher reasoning over Neo4j (Day 10) with Layer 3 Predicate Weighting
  2. Vector search   — Native Neo4j Vector index similarity
"""
from __future__ import annotations

import logging
import math
from typing import NamedTuple
from neo4j import Driver
from google.genai import types

from config import cfg, get_predicate_weight
from rag.router import route_query
from utils.embedder import encode as _encode, chunk_text as _chunk_text
from utils.gemini import get_client

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
    paper_abstracts: tuple[str, ...] = ()
    route: str = "hybrid"


# ── Unified Graph & Vector Retriever ──────────────────────────────────────────

class GraphRetriever:
    _EXTRACTOR_SYSTEM = (
        "You are an entity extractor for a knowledge graph. Extract the core "
        "technical concepts, AI models, or frameworks from the query. Return "
        "ONLY a comma-separated list of terms. Do not include markdown, "
        "explanations, or quotes."
    )

    def __init__(self, driver: Driver):
        self.driver = driver
        self._client = get_client()

    def extract_entities_from_query(self, query: str) -> list[str]:
        """Uses Gemini to intelligently extract entities instead of regex."""
        try:
            response = self._client.models.generate_content(
                model=cfg.LLM_MODEL,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=self._EXTRACTOR_SYSTEM,
                ),
            )
            if response.text:
                entities = [e.strip() for e in response.text.split(",") if e.strip()]
                logger.debug("Gemini extracted entities: %s", entities)
                return entities
            return []
        except Exception as exc:
            logger.warning("Gemini entity extraction failed: %s", exc)
            return []

    def resolve_seed_entities(self, seeds: list[str]) -> list[str]:
        """Resolve raw query seed terms to canonical node names using the
        `entity_name_embedding` vector index.

        This is the retrieval-side counterpart to the ingest-time entity
        resolution in KGMerger._resolve_entity. Substring matching alone misses
        synonym/abbreviation pairs ("LLM" vs "Large Language Model") because they
        share no common substring. Embedding each seed and querying the name
        vector index bridges that gap, finally exercising the index the README
        advertises for dedup.

        Returns the union of the original seeds and every canonical name whose
        name-embedding similarity to a seed is >= SEED_RESOLUTION_THRESHOLD
        (loose, recall-oriented). The originals are kept so the downstream
        substring match still works and nothing that resolved before regresses.
        Best-effort: on any embedding/index failure the originals pass through.
        """
        if not seeds:
            return []

        resolved: dict[str, None] = {s: None for s in seeds}  # ordered set

        try:
            embeddings = _encode(seeds)
        except Exception as exc:
            logger.debug("Seed embedding unavailable, using raw seeds: %s", exc)
            return list(resolved.keys())

        cypher = """
        CALL db.index.vector.queryNodes('entity_name_embedding', $k, $embedding)
        YIELD node, score
        WHERE score >= $threshold
        RETURN node.name AS name, score
        ORDER BY score DESC
        """
        try:
            with self.driver.session() as session:
                for i, seed in enumerate(seeds):
                    records = session.run(
                        cypher,
                        k=cfg.SEED_RESOLUTION_TOP_K,
                        embedding=embeddings[i].tolist(),
                        threshold=cfg.SEED_RESOLUTION_THRESHOLD,
                    ).data()
                    for rec in records:
                        name = rec["name"]
                        if name not in resolved:
                            logger.debug(
                                "Seed resolution (vector): '%s' -> '%s' (sim=%.3f)",
                                seed, name, rec["score"],
                            )
                            resolved[name] = None
        except Exception as exc:
            logger.warning("Vector seed resolution unavailable (%s); using raw seeds", exc)

        return list(resolved.keys())

    def multi_hop_traverse(
        self,
        entity_names: list[str],
        limit: int | None = None,
    ) -> list[GraphTriple]:
        """
        Traverses up to 2 hops across the graph to extract contextual paths,
        incorporating Layer 3 high-signal semantic predicate weighting.
        """
        limit = limit or cfg.GRAPH_TRIPLE_LIMIT
        if not entity_names:
            return []

        # Resolve raw seeds to canonical node names via the name-embedding vector
        # index so abbreviations/synonyms reach the right node. Resolved names
        # are exact node names, so they can be matched exactly (index-backed) in
        # addition to the original loose substring match.
        resolved_seeds = self.resolve_seed_entities(entity_names)

        # Variable length multi-hop matching (1 to 2 hops deep) targeting the custom :RELATION edge type.
        # A node seeds the traversal if it matches a resolved name EXACTLY, or if
        # any raw term is a case-insensitive substring of its name (legacy path).
        cypher = """
        MATCH (seed:Entity)
        WHERE seed.name IN $resolved
           OR ANY(s IN $seeds WHERE toLower(seed.name) CONTAINS toLower(s))
        MATCH path = (seed)-[r*1..2]-(neighbor:Entity)
        WHERE seed <> neighbor
        UNWIND relationships(path) AS rel
        WITH startNode(rel) AS s_node, endNode(rel) AS t_node, rel
        RETURN DISTINCT 
            s_node.name AS subject, 
            coalesce(rel.type, 'related_to') AS predicate,
            t_node.name AS obj,
            coalesce(rel.confidence, 1.0) AS confidence,
            coalesce(rel.mention_count, 1) AS mention_count
        """
        
        with self.driver.session() as session:
            records = session.run(cypher, seeds=entity_names, resolved=resolved_seeds).data()

        processed_triples = []
        for r in records:
            conf = float(r["confidence"])
            mentions = int(r["mention_count"])
            predicate = r["predicate"]
            
            # Layer 3 Scoring Optimization: Confidence * Logarithmic Mentions * High-Signal Predicate Weight
            pred_weight = get_predicate_weight(predicate)
            score = conf * math.log(mentions + 1) * pred_weight
            
            processed_triples.append(
                GraphTriple(
                    subject=r["subject"],
                    predicate=predicate,
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
        """Fetches raw text/abstracts for nodes involved in the multi-hop paths."""
        if not entity_names:
            return []
            
        cypher = """
        MATCH (n:Entity)
        WHERE n.name IN $names AND n.text IS NOT NULL
        RETURN n.name AS name, n.text AS text
        """
        
        try:
            with self.driver.session() as session:
                records = session.run(cypher, names=entity_names).data()
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
        """Queries the native Neo4j chunk vector index for passage-level matches."""
        q_emb = _encode([query])[0].tolist()
        cypher = """
        CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
        YIELD node, score
        OPTIONAL MATCH (node)-[:FROM_DOC]->(doc:Entity)
        RETURN node.text AS text, score, coalesce(doc.name, node.doc_id) AS doc_id
        """
        try:
            with self.driver.session() as s:
                records = s.run(cypher, k=k, embedding=q_emb).data()
            return [
                VectorResult(text=r["text"], score=r["score"], doc_id=r["doc_id"])
                for r in records if r.get("text")
            ]
        except Exception as exc:
            logger.warning("Neo4j chunk vector search failed (is the index built?): %s", exc)
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
            paper_abstracts=tuple(paper_abstracts),
            route=route,
        )

    def index_document(self, doc: dict):
        """Chunk the document, embed each passage, and store :Chunk nodes linked
        to the source :Entity for passage-level semantic retrieval.

        The parent entity keeps its raw `text` (used by summaries and abstract
        fetching), but the retrieval vectors now live on :Chunk nodes — the
        entity's own vector space is reserved for name-based resolution.
        Re-ingesting a document replaces its chunks idempotently.

        Cache invalidation: the cached Gemini `summary` is removed on every
        re-ingest. The summary is derived from the node's text + connected
        relations, both of which can change when a document is re-ingested with
        new content; dropping it forces /api/paper/summary to lazily regenerate
        on next request instead of serving a stale summary forever.
        """
        doc_id = doc.get("title") or doc.get("id", "")
        if not doc_id:
            logger.warning("Cannot index document without a title or ID.")
            return
        abstract_or_text = doc.get("abstract") or doc.get("text", "")
        full_text = f"{doc.get('title', '')}. {abstract_or_text}".strip()
        pub_date = doc.get("published_date") or "Unknown"

        # Persist raw text + publish date on the parent entity (no vector here).
        meta_cypher = """
        MATCH (e:Entity {name: $name})
        SET e.text = $text,
            e.published_date = CASE WHEN $pub_date <> 'Unknown' THEN $pub_date ELSE COALESCE(e.published_date, 'Unknown') END
        REMOVE e.summary
        RETURN count(e) AS matched_count
        """
        with self.driver.session() as s:
            record = s.run(
                meta_cypher, name=doc_id, text=full_text, pub_date=pub_date
            ).single()
            matched = record["matched_count"] if record else 0

        if matched == 0:
            logger.warning("No Neo4j entity found with name '%s' to attach chunks.", doc_id)
            return

        # Build passage chunks and their embeddings.
        chunks = _chunk_text(full_text)
        if not chunks:
            logger.warning("Document '%s' produced no chunks to index.", doc_id)
            return

        try:
            embeddings = _encode(chunks)
        except Exception as exc:
            logger.warning("Chunk embedding failed for '%s': %s", doc_id, exc)
            return

        chunk_rows = [
            {
                "id": f"{doc_id}::chunk::{i}",
                "text": chunk,
                "seq": i,
                "embedding": embeddings[i].tolist(),
            }
            for i, chunk in enumerate(chunks)
        ]

        # Replace any prior chunks for this doc, then recreate — keeps re-ingest idempotent.
        write_cypher = """
        MATCH (e:Entity {name: $name})
        OPTIONAL MATCH (e)<-[:FROM_DOC]-(old:Chunk)
        DETACH DELETE old
        WITH e
        UNWIND $chunks AS chunk
        CREATE (c:Chunk {
            id: chunk.id,
            text: chunk.text,
            seq: chunk.seq,
            doc_id: $name,
            embedding: chunk.embedding
        })
        MERGE (c)-[:FROM_DOC]->(e)
        RETURN count(c) AS created
        """
        with self.driver.session() as s:
            result = s.run(write_cypher, name=doc_id, chunks=chunk_rows).single()
            created = result["created"] if result else 0
        logger.info("Indexed %d chunk(s) for node: %s", created, doc_id)