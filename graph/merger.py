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
from utils.embedder import encode as _encode

logger = logging.getLogger(__name__)


class KGMerger:
    """
    Merges entities and triples into Neo4j with automated vector indexing and clustering.
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
        properties = properties or {}
        
        # Ensure embedding is generated and stored for vector indexing and clustering
        if "embedding" not in properties:
            try:
                properties["embedding"] = _encode(name)[0].tolist()
            except Exception as exc:
                logger.debug("Entity embedding unavailable for '%s': %s", name, exc)

        canonical = self._resolve_entity(name, entity_type)

        cypher = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET 
            e.type = $type,
            e.created = timestamp(), 
            e.mention_count = 1
        ON MATCH SET 
            e.type = COALESCE(e.type, $type),
            e.updated = timestamp(), 
            e.mention_count = coalesce(e.mention_count, 0) + 1
        SET e += $props
        RETURN e.name AS name
        """
        with self.driver.session() as s:
            result = s.run(cypher, name=canonical, type=entity_type, props=properties)
            record = result.single()
            return record["name"] if record else canonical

    def _resolve_entity(self, name: str, entity_type: str) -> str:
        # 1. Check for an exact match first (fastest)
        with self.driver.session() as s:
            exact = s.run(
                "MATCH (e:Entity {name: $name, type: $type}) RETURN e.name LIMIT 1",
                name=name, type=entity_type,
            ).single()
            if exact:
                return name

        # 2. Generate embedding for the new entity name using your embedder
        try:
            query_emb = _encode(name)[0].tolist()
        except Exception as exc:
            logger.debug("Entity embedding unavailable for '%s': %s", name, exc)
            return name

        # 3. Use Neo4j native vector index to find the most similar entity of the same type
        cypher = """
        CALL db.index.vector.queryNodes('entity_embedding', 1, $embedding)
        YIELD node, score
        WHERE node.type = $type AND score >= $threshold
        RETURN node.name AS name, score
        """

        try:
            with self.driver.session() as s:
                result = s.run(
                    cypher,
                    embedding=query_emb,
                    type=entity_type,
                    threshold=cfg.ENTITY_MERGE_THRESHOLD,
                ).single()
        except Exception as exc:
            logger.warning("Neo4j vector lookup unavailable for '%s': %s", name, exc)
            return name

        if result:
            canonical = result["name"]
            score = result["score"]
            logger.debug("Entity merge (Neo4j Vector): '%s' → '%s' (sim=%.3f)", name, canonical, score)
            return canonical

        return name

    # ── Triple upsert ─────────────────────────────────────────────────────────

    # ── Triple upsert ─────────────────────────────────────────────────────────

    def upsert_triple(
        self,
        subj: str,
        pred: str,
        obj: str,
        source_paper: str,
        confidence: float = 1.0,
        subj_type: str = "Unknown",
        obj_type: str = "Unknown",
        subj_props: dict | None = None,
        obj_props: dict | None = None,
    ) -> None:
        # Ensure both entities are fully upserted with their granular types and property payloads
        self.upsert_entity(subj, subj_type, subj_props)
        self.upsert_entity(obj, obj_type, obj_props)

        cypher = """
        MATCH (s:Entity {name: $subj})
        MATCH (o:Entity {name: $obj})
        MERGE (s)-[r:RELATION {type: $pred}]->(o) 
        ON CREATE SET
            r.sources       = [$src],
            r.confidence    = $conf,
            r.first_seen    = timestamp(),
            r.last_seen     = timestamp(),
            r.mention_count = 1 
        ON MATCH SET 
            r.sources       = CASE WHEN $src IN r.sources THEN r.sources ELSE r.sources + $src END,
            r.confidence    = (r.confidence * r.mention_count + $conf) / (r.mention_count + 1),
            r.last_seen     = timestamp(),
            r.mention_count = r.mention_count + 1
        """
        with self.driver.session() as session:
            session.run(
                cypher,
                subj=subj,
                obj=obj,
                pred=pred,
                src=source_paper,
                conf=confidence,
            )
    # ── Vector Similarity Clustering ──────────────────────────────────────────

    def run_vector_clustering_job(self, threshold: float | None = None) -> int:
        """
        Scans the entity embedding index periodically to bridge concept silos
        and merge semantic duplicates using APOC refactoring.
        """
        threshold = threshold or cfg.ENTITY_MERGE_THRESHOLD
        query = """
        MATCH (e1:Entity)
        WHERE e1.embedding IS NOT NULL
        CALL db.index.vector.queryNodes('entity_embedding', 5, e1.embedding)
        YIELD node AS e2, score
        WHERE e1 <> e2 AND e1.type = e2.type AND score >= $threshold
        RETURN e1.name AS id1, e2.name AS id2, score
        """
        merged_count = 0
        with self.driver.session() as s:
            results = s.run(query, threshold=threshold)
            for record in results:
                name1, name2, score = record["id1"], record["id2"], record["score"]
                if name1 != name2:
                    logger.debug("Cluster merging: '%s' and '%s' (sim=%.3f)", name1, name2, score)
                    self._merge_entities(name1, name2)
                    merged_count += 1
        return merged_count

    def _merge_entities(self, primary_name: str, duplicate_name: str):
        cypher = """
        MATCH (p:Entity {name: $primary})
        MATCH (d:Entity {name: $duplicate})
        WHERE p <> d
        CALL apoc.refactor.mergeNodes([p, d], {properties: 'combine', mergeRels: true})
        YIELD node
        RETURN node.name
        """
        with self.driver.session() as s:
            try:
                s.run(cypher, primary=primary_name, duplicate=duplicate_name)
            except Exception as exc:
                logger.warning("Failed to merge nodes '%s' and '%s': %s", primary_name, duplicate_name, exc)

    # ── Bulk operations ───────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return node, relationship, and vector counts."""
        with self.driver.session() as s:
            nodes = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rels = s.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS c").single()["c"]
            papers = s.run("MATCH (n:Entity) WHERE n.type IN ['Paper', 'Document'] RETURN count(n) AS c").single()["c"]
            vectors = s.run("MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN count(n) AS c").single()["c"]
            
        return {"entities": nodes, "relations": rels, "papers": papers, "vectors": vectors}