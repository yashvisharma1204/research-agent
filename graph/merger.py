"""
graph/merger.py
Responsible for upserting entities and triples into Neo4j.
Key feature: fuzzy entity resolution prevents duplicate nodes
when the same real-world entity appears under slightly different names.
"""
from __future__ import annotations

import logging
import hashlib
import numpy as np
from neo4j import Driver

from config import cfg
from utils.embedder import encode as _encode

logger = logging.getLogger(__name__)

# Ontology types that are promoted to real Neo4j labels (in addition to the
# base :Entity label). Kept as a whitelist so a hallucinated / malformed type
# from the LLM can never inject an arbitrary label. Must mirror the type list
# in ingestion/extractors.py.
ONTOLOGY_LABELS = {
    "Model",
    "ArchitecturalInnovation",
    "TrainingObjective",
    "LossFunction",
    "HardwareConstraint",
    "Dataset",
    "Metric",
    "Task",
    "Concept",
    "Author",
    "Organisation",
    "Paper",
    "Document",
    "Result",
    "Contribution",
}


def _type_to_label(entity_type: str | None) -> str | None:
    """Map an ontology *type* string to a safe Neo4j label, or None.

    Only whitelisted types become labels; anything else (including 'Unknown')
    returns None so no extra label is applied.
    """
    if not entity_type:
        return None
    label = entity_type.strip()
    return label if label in ONTOLOGY_LABELS else None


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
        embedding: list[float] | None = None,
    ) -> str:
        """Upsert a single entity.

        *embedding* is an optional precomputed name embedding (as a plain list of
        floats). Passing it avoids re-encoding the name — used by the batched
        document path so every unique name is encoded at most once per document.
        """
        props = dict(properties) if properties else {}
        canonical = self._resolve_entity(name, entity_type, embedding=embedding) or name
        # Name-only embedding used exclusively for entity resolution / clustering.
        # Retrieval embeddings live on :Chunk nodes, not here — keeping them apart
        # prevents name-similarity and passage-similarity from polluting each other.
        if "name_embedding" not in props:
            try:
                # Reuse the precomputed vector when we resolved back to the same
                # name; otherwise encode the canonical form once.
                if embedding is not None and canonical == name:
                    props["name_embedding"] = embedding
                else:
                    props["name_embedding"] = _encode([canonical])[0].tolist()
            except Exception as exc:
                logger.debug("Entity name embedding unavailable for '%s': %s", canonical, exc)

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
        WITH e
        CALL apoc.create.addLabels(e, $labels) YIELD node
        RETURN node.name AS name
        """
        # Promote the ontology type to a real label (in addition to :Entity).
        # Whitelisted only, so a bad LLM type can't inject an arbitrary label.
        label = _type_to_label(entity_type)
        labels = [label] if label else []
        with self.driver.session() as s:
            try:
                result = s.run(
                    cypher, name=canonical, type=entity_type, props=props, labels=labels
                )
                record = result.single()
                return record["name"] if record else canonical
            except Exception as exc:
                # APOC unavailable / label call failed — fall back to a plain
                # upsert without the dynamic label so ingestion still succeeds.
                logger.warning("addLabels failed for '%s' (%s); upserting without type label", canonical, exc)
                fallback = """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = $type, e.created = timestamp(), e.mention_count = 1
                ON MATCH SET e.type = COALESCE(e.type, $type), e.updated = timestamp(),
                    e.mention_count = coalesce(e.mention_count, 0) + 1
                SET e += $props
                RETURN e.name AS name
                """
                record = s.run(fallback, name=canonical, type=entity_type, props=props).single()
                return record["name"] if record else canonical

    def _resolve_entity(
        self,
        name: str,
        entity_type: str,
        embedding: list[float] | None = None,
    ) -> str:
        # 1. Check for an exact match first (fastest)
        with self.driver.session() as s:
            exact = s.run(
                "MATCH (e:Entity {name: $name, type: $type}) RETURN e.name LIMIT 1",
                name=name, type=entity_type,
            ).single()
            if exact:
                return name

        # 2. Generate embedding for the new entity name (reuse precomputed if given)
        try:
            query_emb = embedding if embedding is not None else _encode([name])[0].tolist()
        except Exception as exc:
            logger.debug("Entity embedding unavailable for '%s': %s", name, exc)
            return name

        # 3. Use Neo4j native vector index to find the most similar entity of the same type
        cypher = """
        CALL db.index.vector.queryNodes('entity_name_embedding', 10, $embedding)
        YIELD node, score
        WHERE node.type = $type AND score >= $threshold
        RETURN node.name AS name, score
        ORDER BY score DESC
        LIMIT 1
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
        skip_entity_upsert: bool = False,
    ) -> None:
        # Ensure both entities are fully upserted with their granular types and
        # property payloads. The batched document path sets skip_entity_upsert=True
        # because it has already upserted every unique entity once.
        if not skip_entity_upsert:
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

    # ── Batched document upsert ───────────────────────────────────────────────

    def upsert_document_triples(self, triples: list, source_id: str) -> int:
        """Upsert all triples of one document with batched embedding + dedup.

        Latency optimisation over calling upsert_entity/upsert_triple per triple:
          1. Collect every distinct (name, type) mention across the document.
          2. Batch-encode all unique names in a SINGLE embedder call.
          3. Dedup exact-string duplicates (safe; same as the exact-match path).
          4. Dedup intra-document *fuzzy* duplicates greedily using the batched
             embeddings + ENTITY_MERGE_THRESHOLD, so two brand-new near-duplicate
             names in the same document collapse to one node — preserving the
             behaviour of the old incremental one-at-a-time path.
          5. Upsert each surviving unique entity once (reusing its embedding).
          6. Write triples via a name->canonical map, skipping per-triple entity
             upserts.

        Returns the number of triples written.
        """
        if not triples:
            return 0

        # 1. Collect distinct mentions. First mention wins for type/props.
        #    Key on (name, type) since resolution is type-scoped.
        mentions: dict[tuple[str, str], dict] = {}
        for t in triples:
            for name, etype, props in (
                (t.subject, t.subject_type, t.subject_props),
                (t.obj, t.obj_type, t.obj_props),
            ):
                key = (name, etype)
                if key not in mentions:
                    mentions[key] = dict(props) if props else {}

        unique_keys = list(mentions.keys())
        names = [k[0] for k in unique_keys]

        # 2. Batch-encode every unique name in one forward pass.
        embeddings: list[list[float] | None] = [None] * len(names)
        try:
            matrix = _encode(names)
            embeddings = [row.tolist() for row in matrix]
        except Exception as exc:
            logger.debug("Batch name embedding unavailable for document '%s': %s", source_id, exc)

        # 3 + 4. Build canonical map with exact + intra-document fuzzy dedup.
        #        canonical_map: (name, type) -> canonical name to write with.
        canonical_map: dict[tuple[str, str], str] = {}
        # Per type, remember accepted (name, embedding) reps for fuzzy compare.
        reps_by_type: dict[str, list[tuple[str, np.ndarray]]] = {}
        # Embedding to write for each surviving canonical entity.
        emb_for_key: dict[tuple[str, str], list[float] | None] = {}
        threshold = cfg.ENTITY_MERGE_THRESHOLD

        for idx, key in enumerate(unique_keys):
            name, etype = key
            emb = embeddings[idx]

            reps = reps_by_type.setdefault(etype, [])

            # Exact-string dedup: identical (name, type) already collapsed by the
            # dict, so any remaining same-name is a different type — no action.

            # Fuzzy dedup against reps already accepted for this type.
            matched = None
            if emb is not None:
                emb_vec = np.asarray(emb, dtype="float32")
                best_score = -1.0
                for rep_name, rep_vec in reps:
                    # Vectors are L2-normalised by the embedder, so dot == cosine.
                    score = float(np.dot(emb_vec, rep_vec))
                    if score >= threshold and score > best_score:
                        best_score = score
                        matched = rep_name
                if matched is not None:
                    logger.debug(
                        "Intra-doc fuzzy merge: '%s' -> '%s' (sim=%.3f)",
                        name, matched, best_score,
                    )

            if matched is not None:
                canonical_map[key] = matched
            else:
                canonical_map[key] = name
                if emb is not None:
                    reps.append((name, np.asarray(emb, dtype="float32")))
                emb_for_key[key] = emb

        # 5. Upsert each surviving unique entity once, reusing its embedding.
        #    A resolved canonical (matched against the *existing* graph) may
        #    differ from the in-memory name, so track the final written name and
        #    map every alias key to it. Process canonical keys before aliases so
        #    the canonical is always upserted first.
        resolved_name: dict[tuple[str, str], str] = {}

        # Upsert canonicals (keys that map to themselves).
        for key in unique_keys:
            if canonical_map[key] == key[0]:
                final = self.upsert_entity(
                    key[0], key[1], mentions[key],
                    embedding=emb_for_key.get(key),
                )
                resolved_name[key] = final

        # Map aliases (fuzzy-merged names) onto their canonical's final name.
        for key in unique_keys:
            if key not in resolved_name:
                canonical_key = (canonical_map[key], key[1])
                resolved_name[key] = resolved_name.get(canonical_key, canonical_map[key])

        # 6. Write triples using the resolved names; skip per-triple entity upsert.
        written = 0
        for t in triples:
            subj = resolved_name.get((t.subject, t.subject_type), t.subject)
            obj = resolved_name.get((t.obj, t.obj_type), t.obj)
            self.upsert_triple(
                subj, t.predicate, obj, source_id, t.confidence,
                subj_type=t.subject_type, obj_type=t.obj_type,
                skip_entity_upsert=True,
            )
            written += 1
        return written

    # ── First-class Paper + literature subgraph ───────────────────────────────

    def upsert_paper(self, doc: dict) -> str:
        """Upsert a paper as a first-class :Paper node with rich metadata, then
        attach its authors and citation edges.

        doc keys used: title, url, source, citations, published_date, abstract,
        venue, year, authors (list[str]), references (list[str] of cited titles).
        Returns the canonical paper name.
        """
        title = doc.get("title", "")
        if not title:
            return ""

        pub_date = doc.get("published_date") or "Unknown"
        authors = [a.strip() for a in (doc.get("authors") or []) if a and a.strip()]
        props = {
            "url": doc.get("url", ""),
            "source": doc.get("source", ""),
            "citations": doc.get("citations", 0) or 0,
            "published_date": pub_date,
            "venue": doc.get("venue", "") or "",
            "year": str(doc.get("year", "") or ""),
            "text": doc.get("abstract", "") or "",
            # Authors live here as lightweight provenance, NOT as embedded
            # :Author entities (which flood the graph + vector index). Still
            # fully queryable via p.authors. See cfg.EXTRACT_AUTHORS.
            "authors": authors,
        }
        paper_name = self.upsert_entity(title, "Paper", props)

        # Only build the embedded author/citation subgraph when explicitly
        # enabled. By default authors are the `authors` property set above.
        if cfg.EXTRACT_AUTHORS:
            self.upsert_authors(paper_name, authors)
        self.upsert_citations(paper_name, doc.get("references") or [])
        return paper_name

    def upsert_authors(self, paper_name: str, authors: list[str]) -> int:
        """Create (:Author)-[:AUTHORED]->(:Paper) edges. Authors become real
        :Author-labelled entities via the standard entity upsert path."""
        written = 0
        for author in authors:
            author = (author or "").strip()
            if not author:
                continue
            # Explicit empty name_embedding keeps authors OUT of the entity
            # vector index (they're metadata, not retrievable research content),
            # while still letting the AUTHORED citation subgraph exist.
            canonical = self.upsert_entity(author, "Author", {"name_embedding": None})
            cypher = """
            MATCH (a:Entity {name: $author})
            MATCH (p:Entity {name: $paper})
            MERGE (a)-[:AUTHORED]->(p)
            """
            with self.driver.session() as s:
                try:
                    s.run(cypher, author=canonical, paper=paper_name)
                    written += 1
                except Exception as exc:
                    logger.debug("AUTHORED edge failed (%s -> %s): %s", canonical, paper_name, exc)
        return written

    def upsert_citations(self, paper_name: str, cited_titles: list[str]) -> int:
        """Create (:Paper)-[:CITES]->(:Paper) edges. Cited papers are upserted as
        lightweight :Paper stubs (title only) so the citation network is
        navigable even before the cited paper is fully ingested."""
        written = 0
        for cited in cited_titles:
            cited = (cited or "").strip()
            if not cited or cited == paper_name:
                continue
            canonical = self.upsert_entity(cited, "Paper")
            cypher = """
            MATCH (src:Entity {name: $src})
            MATCH (dst:Entity {name: $dst})
            WHERE src <> dst
            MERGE (src)-[:CITES]->(dst)
            """
            with self.driver.session() as s:
                try:
                    s.run(cypher, src=paper_name, dst=canonical)
                    written += 1
                except Exception as exc:
                    logger.debug("CITES edge failed (%s -> %s): %s", paper_name, canonical, exc)
        return written

    # ── Results reification ───────────────────────────────────────────────────

    def upsert_results(self, results: list, source_id: str) -> int:
        """Write reified benchmark results as an evidence subgraph:

            (:Model)-[:ACHIEVES]->(:Result {value, source})-[:ON]->(:Dataset)
            (:Result)-[:MEASURED_BY]->(:Metric)

        Each *results* item is a ScoreResult (model/dataset/metric/value). The
        Model/Dataset/Metric endpoints are upserted through the normal entity
        path so they get their type labels + name embeddings. The Result node is
        a reified relationship carrying the numeric evidence and provenance.
        Returns the number of results written.
        """
        written = 0
        for r in results or []:
            model = getattr(r, "model", "").strip()
            dataset = getattr(r, "dataset", "").strip()
            metric = getattr(r, "metric", "").strip()
            value = getattr(r, "value", "") or ""
            if not (model and dataset and metric):
                continue

            model_c = self.upsert_entity(model, "Model")
            dataset_c = self.upsert_entity(dataset, "Dataset")
            metric_c = self.upsert_entity(metric, "Metric")

            # Deterministic result id so re-ingesting the same claim is idempotent.
            result_id = f"{model_c}::{dataset_c}::{metric_c}"
            cypher = """
            MATCH (m:Entity {name: $model})
            MATCH (d:Entity {name: $dataset})
            MATCH (met:Entity {name: $metric})
            MERGE (res:Result {id: $rid})
              ON CREATE SET res.value = $value, res.source = $src, res.created = timestamp()
              ON MATCH  SET res.value = $value, res.updated = timestamp()
            MERGE (m)-[:ACHIEVES]->(res)
            MERGE (res)-[:ON]->(d)
            MERGE (res)-[:MEASURED_BY]->(met)
            """
            with self.driver.session() as s:
                try:
                    s.run(
                        cypher,
                        model=model_c, dataset=dataset_c, metric=metric_c,
                        rid=result_id, value=str(value), src=source_id,
                    )
                    written += 1
                except Exception as exc:
                    logger.debug("Result reification failed (%s): %s", result_id, exc)
        return written

    # ── Contribution reification ──────────────────────────────────────────────

    def upsert_contributions(self, contributions: list, source_id: str,
                             paper_name: str | None = None) -> int:
        """Write reified paper contributions as a navigable subgraph:

            (:Paper)-[:RELATION {type:'makes_claim'}]->(:Contribution)
            (:Contribution)-[:RELATION {type:'addresses'}]->(target)   # problem
            (:Contribution)-[:RELATION {type:'via'}]->(mechanism)      # solution

        Design notes:
        * The Contribution is a first-class :Entity:Contribution node so it takes
          part in the retriever's generic (:Entity)-[:RELATION*1..2]-(:Entity)
          traversal with ZERO retriever changes.
        * Edges use the existing :RELATION {type: ...} pattern, and the predicate
          types (makes_claim/addresses/via) are registered as HIGH_SIGNAL so they
          outrank `related_to` noise in traversal scoring.
        * The Contribution carries an explicit null name_embedding so these
          reified claim nodes never pollute the entity name-resolution vector
          index (same discipline as demoted authors).

        *contributions* items are Contribution objects. *source_id* is the
        provenance id stored on edges/nodes. *paper_name* is the :Paper node's
        name (its title) that the makes_claim edge originates from; it differs
        from source_id because the Paper node is keyed on title while source_id
        is the document id. When paper_name is None the makes_claim edge is
        skipped (contribution still written, just not linked to a paper).
        Returns the number of contributions written.
        """
        written = 0
        for c in contributions or []:
            statement = getattr(c, "statement", "").strip()
            if not statement:
                continue
            kind = c.normalised_kind() if hasattr(c, "normalised_kind") else "proposes_method"

            # Deterministic id → re-ingesting the same claim is idempotent.
            # Uses a stable hash (not Python's per-process-salted hash()).
            digest = hashlib.sha1(statement.encode("utf-8")).hexdigest()[:12]
            contrib_id = f"{source_id}::contrib::{digest}"

            # The Contribution node itself (reified, no name embedding).
            self.upsert_entity(
                contrib_id, "Contribution",
                {"statement": statement, "kind": kind, "source": source_id,
                 "name_embedding": None},
            )

            # (:Paper)-[:makes_claim]->(:Contribution) — only when we know the
            # paper node name. The paper node is upserted by upsert_paper first.
            if paper_name:
                self.upsert_triple(
                    paper_name, "makes_claim", contrib_id, source_id,
                    confidence=1.0, subj_type="Paper", obj_type="Contribution",
                    skip_entity_upsert=True,
                )

            # (:Contribution)-[:addresses]->(problem)
            if getattr(c, "addresses", "").strip():
                self.upsert_entity(c.addresses, c.addresses_type or "Concept")
                self.upsert_triple(
                    contrib_id, "addresses", c.addresses, source_id,
                    confidence=1.0, subj_type="Contribution",
                    obj_type=c.addresses_type or "Concept",
                    skip_entity_upsert=True,
                )

            # (:Contribution)-[:via]->(mechanism)
            if getattr(c, "via", "").strip():
                self.upsert_entity(c.via, c.via_type or "ArchitecturalInnovation")
                self.upsert_triple(
                    contrib_id, "via", c.via, source_id,
                    confidence=1.0, subj_type="Contribution",
                    obj_type=c.via_type or "ArchitecturalInnovation",
                    skip_entity_upsert=True,
                )
            written += 1
        return written

    # ── Vector Similarity Clustering ──────────────────────────────────────────

    def run_vector_clustering_job(self, threshold: float | None = None) -> int:
        """
        Scans the entity embedding index periodically to bridge concept silos
        and merge semantic duplicates using APOC refactoring.
        """
        """
        Scans the entity embedding index periodically to bridge concept silos
        and merge semantic duplicates using APOC refactoring.
        """
        threshold = threshold or cfg.ENTITY_MERGE_THRESHOLD
        query = """
        MATCH (e1:Entity)
        WHERE e1.name_embedding IS NOT NULL
        CALL db.index.vector.queryNodes('entity_name_embedding', 5, e1.name_embedding)
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
            # Count papers by real :Paper label OR legacy type string (pre-migration nodes).
            papers = s.run(
                "MATCH (n:Entity) WHERE n:Paper OR n.type IN ['Paper', 'Document'] "
                "RETURN count(n) AS c"
            ).single()["c"]
            vectors = s.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS c").single()["c"]
            authors = s.run("MATCH (n:Author) RETURN count(n) AS c").single()["c"]
            results = s.run("MATCH (r:Result) RETURN count(r) AS c").single()["c"]
            contributions = s.run("MATCH (c:Contribution) RETURN count(c) AS c").single()["c"]

        return {
            "entities": nodes,
            "relations": rels,
            "papers": papers,
            "vectors": vectors,
            "authors": authors,
            "results": results,
            "contributions": contributions,
        }