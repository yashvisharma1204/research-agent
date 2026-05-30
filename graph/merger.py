"""
graph/merger.py
Stores entities, triples, AND full paper nodes with their
concept/method/dataset connections and paper-to-paper relationships.
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
        from utils.embedder import encode
        _embed_model = encode
    return _embed_model


class KGMerger:

    def __init__(self, driver: Driver):
        self.driver = driver

    # ── Entity upsert ─────────────────────────────────────────────────────────

    def upsert_entity(self, name: str, entity_type: str, properties: dict | None = None) -> str:
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
        with self.driver.session() as s:
            exact = s.run("MATCH (e:Entity {name: $name, type: $type}) RETURN e.name LIMIT 1",
                          name=name, type=entity_type).single()
            if exact: return name

        with self.driver.session() as s:
            candidates = s.run("MATCH (e:Entity {type: $type}) RETURN e.name AS name LIMIT 500",
                               type=entity_type).data()
        if not candidates: return name

        candidate_names = [c["name"] for c in candidates]
        encode = _get_embed_model()
        query_emb = encode([name])[0]
        cand_embs = encode(candidate_names)
        sims = cand_embs @ query_emb
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= cfg.ENTITY_MERGE_THRESHOLD:
            canonical = candidate_names[best_idx]
            logger.debug("Entity merge: '%s' → '%s' (sim=%.3f)", name, canonical, sims[best_idx])
            return canonical
        return name

    # ── Triple upsert ─────────────────────────────────────────────────────────

    def upsert_triple(self, subj: str, pred: str, obj: str,
                      source_paper: str, confidence: float = 1.0) -> None:
        cypher = """
        MERGE (s:Entity {name: $subj}) ON CREATE SET s.type = 'Unknown', s.created = timestamp()
        MERGE (o:Entity {name: $obj}) ON CREATE SET o.type = 'Unknown', o.created = timestamp()
        MERGE (s)-[r:RELATION {type: $pred}]->(o) ON CREATE SET
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
        with self.driver.session() as s:
            s.run(cypher, subj=subj, obj=obj, pred=pred, src=source_paper, conf=confidence)

    # ── Paper node upsert ─────────────────────────────────────────────────────

    def upsert_paper_node(self, paper) -> None:
        """
        Store a full PaperNode in Neo4j:
        - Creates/updates a Paper node
        - Links it to every concept, method, dataset, metric it contains
        - Links it to other papers it builds on
        """
        from ingestion.extractors import PaperNode

        # 1. Upsert the Paper node itself
        cypher_paper = """
        MERGE (p:Paper {title: $title})
        ON CREATE SET
            p.paper_id   = $paper_id,
            p.year       = $year,
            p.url        = $url,
            p.citations  = $citations,
            p.difficulty = $difficulty,
            p.summary    = $summary,
            p.authors    = $authors,
            p.created    = timestamp()
        ON MATCH SET
            p.citations  = $citations,
            p.url        = $url,
            p.updated    = timestamp()
        """
        with self.driver.session() as s:
            s.run(cypher_paper,
                  title=paper.title,
                  paper_id=paper.paper_id,
                  year=paper.year,
                  url=paper.url,
                  citations=paper.citations,
                  difficulty=paper.difficulty,
                  summary=paper.one_line_summary,
                  authors=paper.authors)

        # 2. Link paper → concepts
        self._link_paper_to_entities(paper.title, paper.concepts, "Concept", "introduces")

        # 3. Link paper → methods
        self._link_paper_to_entities(paper.title, paper.methods, "Method", "proposes")

        # 4. Link paper → datasets
        self._link_paper_to_entities(paper.title, paper.datasets, "Dataset", "uses_dataset")

        # 5. Link paper → metrics
        self._link_paper_to_entities(paper.title, paper.metrics, "Metric", "evaluates_with")

        # 6. Link paper → papers it builds on
        for ref_title in paper.builds_on:
            if ref_title.strip():
                self._link_paper_to_paper(paper.title, ref_title.strip(), "builds_on")

        # 7. Also store key findings as special entity
        if paper.key_findings:
            findings_text = " | ".join(paper.key_findings[:3])
            self._link_paper_to_entities(paper.title, [findings_text], "Concept", "contributes")

        logger.info("Paper node stored: '%s' (%d concepts, %d methods, %d builds_on)",
                    paper.title, len(paper.concepts), len(paper.methods), len(paper.builds_on))

    def _link_paper_to_entities(self, paper_title: str, entity_names: list[str],
                                 entity_type: str, predicate: str) -> None:
        """Create Paper -[predicate]-> Entity edges."""
        for name in entity_names:
            if not name or not name.strip():
                continue
            cypher = """
            MATCH (p:Paper {title: $paper_title})
            MERGE (e:Entity {name: $name, type: $etype})
                ON CREATE SET e.created = timestamp(), e.mention_count = 1
                ON MATCH  SET e.mention_count = coalesce(e.mention_count, 0) + 1
            MERGE (p)-[r:RELATION {type: $pred}]->(e)
                ON CREATE SET r.created = timestamp(), r.confidence = 1.0, r.mention_count = 1
                ON MATCH  SET r.mention_count = r.mention_count + 1
            """
            try:
                with self.driver.session() as s:
                    s.run(cypher, paper_title=paper_title, name=name.strip(),
                          etype=entity_type, pred=predicate)
            except Exception as exc:
                logger.warning("Failed to link paper '%s' → '%s': %s", paper_title, name, exc)

    def _link_paper_to_paper(self, from_title: str, to_title: str, predicate: str) -> None:
        """Create Paper -[builds_on]-> Paper edges."""
        cypher = """
        MERGE (p1:Paper {title: $from_title})
        MERGE (p2:Paper {title: $to_title})
        MERGE (p1)-[r:RELATION {type: $pred}]->(p2)
            ON CREATE SET r.created = timestamp(), r.confidence = 1.0
        """
        try:
            with self.driver.session() as s:
                s.run(cypher, from_title=from_title, to_title=to_title, pred=predicate)
        except Exception as exc:
            logger.warning("Failed paper→paper link '%s'→'%s': %s", from_title, to_title, exc)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self.driver.session() as s:
            nodes  = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rels   = s.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS c").single()["c"]
            papers = s.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
        return {"entities": nodes, "relations": rels, "papers": papers}