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

from config import cfg

logger = logging.getLogger(__name__)

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

class KGMerger:
    """
    Merges entities and triples into Neo4j.

    Entity resolution strategy:
    1. Exact name+type match → reuse existing node (via MERGE)
    2. Fuzzy embedding similarity above threshold → merge into existing node
    3. No match → create new node
    """

    def __init__(self, driver: Driver):
        self.driver = driver

    # ── Entity upsert ─────────────────────────────────────────────────────────

    def upsert_entity(
        self,
        name: str,
        entity_type: str,
        properties: dict | None = None,
    ) -> str:
        """
        Insert or update an entity node.
        Returns the canonical name used in the graph.
        """
        properties = properties or {}
        canonical = self._resolve_entity(name, entity_type)

        cypher = """
        MERGE (e:Entity {name: $name, type: $type})
        ON CREATE SET e += $props, e.created = timestamp(), e.mention_count = 1
        ON MATCH  SET e += $props, e.updated = timestamp(), e.mention_count = coalesce(e.mention_count, 0) + 1
        RETURN e.name AS name
        """
        with self.driver.session() as s:
            result = s.run(cypher, name=canonical, type=entity_type, props=properties)
            record = result.single()
            return record["name"] if record else canonical

    def _resolve_entity(self, name: str, entity_type: str) -> str:
        """
        Look for an existing entity that is semantically equivalent to *name*.
        Returns the canonical name to use (existing or new).
        """
        # Quick exact check first (avoids loading embedding model unnecessarily)
        with self.driver.session() as s:
            exact = s.run(
                "MATCH (e:Entity {name: $name, type: $type}) RETURN e.name LIMIT 1",
                name=name, type=entity_type,
            ).single()
            if exact:
                return name   # already exists verbatim

        # Fuzzy check via sentence embeddings
        with self.driver.session() as s:
            candidates = s.run(
                "MATCH (e:Entity {type: $type}) RETURN e.name AS name LIMIT 500",
                type=entity_type,
            ).data()

        if not candidates:
            return name

        candidate_names = [c["name"] for c in candidates]
        model = _get_embed_model()
        query_emb = model.encode(name, normalize_embeddings=True)
        cand_embs = model.encode(candidate_names, normalize_embeddings=True)

        sims = cand_embs @ query_emb   # cosine similarity (normalized)
        best_idx = int(np.argmax(sims))

        if sims[best_idx] >= cfg.ENTITY_MERGE_THRESHOLD:
            canonical = candidate_names[best_idx]
            logger.debug(
                "Entity merge: '%s' → '%s' (sim=%.3f)", name, canonical, sims[best_idx]
            )
            return canonical

        return name   # new entity

    # ── Triple upsert ─────────────────────────────────────────────────────────

    def upsert_triple(
        self,
        subj: str,
        pred: str,
        obj: str,
        source_paper: str,
        confidence: float = 1.0,
    ) -> None:
        """
        Merge a (subject)-[predicate]->(object) triple.
        - Confidence is averaged across all sources.
        - Source list grows with each new paper confirming the relationship.
        """
        cypher = """
        MERGE (s:Entity {name: $subj}) ON CREATE SET s.type = 'unknown', s.created = timestamp()
        MERGE (o:Entity {name: $obj}) ON CREATE SET o.type = 'unknown', o.created = timestamp()
        MERGE (s)-[r:RELATION {type: $pred}]->(o) ON CREATE SET
            r.sources       = [$src],
            r.confidence    = $conf,
            r.first_seen    = timestamp(),
            r.last_seen     = timestamp(),
            r.mention_count = 1 ON MATCH SET
            r.sources       = CASE WHEN $src IN r.sources
                                THEN r.sources
                                ELSE r.sources + $src END,
            r.confidence    = (r.confidence * r.mention_count + $conf)/ (r.mention_count + 1),
            r.last_seen     = timestamp(),
            r.mention_count = r.mention_count + 1
        """
        with self.driver.session() as s:
            s.run(cypher, subj=subj, obj=obj, pred=pred, src=source_paper, conf=confidence)

    # ── Bulk operations ───────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return node and relationship counts."""
        with self.driver.session() as s:
            nodes = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rels = s.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS c").single()["c"]
            papers = s.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
        return {"entities": nodes, "relations": rels, "papers": papers}
