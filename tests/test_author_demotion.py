"""
tests/test_author_demotion.py

Verifies the author-demotion behaviour (Stage A of the graph-quality cleanup):

  * By default (cfg.EXTRACT_AUTHORS == False) a paper's authors are stored as a
    plain `authors` list PROPERTY on the :Paper node — NOT as embedded :Author
    entities, and NO :AUTHORED edges are written.
  * Setting cfg.EXTRACT_AUTHORS = True restores the AUTHORED subgraph.
  * The extractor quality filter drops Author-typed triples when the flag is off
    and keeps them when it is on.
"""
import numpy as np
import pytest

from graph.merger import KGMerger


class _PaperRecordingDriver:
    """Fake Neo4j driver that records entity upserts (name + props), AUTHORED
    edges, and CITES edges. All resolution lookups miss so names pass through."""

    def __init__(self):
        self.entities = []   # (name, props)
        self.authored = []   # (author, paper)
        self.cites = []      # (src, dst)

    def session(self):
        driver = self

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def run(self, cypher, **kw):
                c = cypher.strip()
                if c.startswith("MATCH (e:Entity {name:"):
                    return type("R", (), {"single": lambda self: None})()
                if "db.index.vector.queryNodes" in cypher:
                    return type("R", (), {"single": lambda self: None})()
                if c.startswith("MERGE (e:Entity"):
                    driver.entities.append((kw["name"], kw.get("props", {})))
                    name = kw["name"]
                    return type("R", (), {"single": lambda self: {"name": name}})()
                if "MERGE (a)-[:AUTHORED]->(p)" in cypher:
                    driver.authored.append((kw["author"], kw["paper"]))
                    return type("R", (), {"single": lambda self: None})()
                if "MERGE (src)-[:CITES]->(dst)" in cypher:
                    driver.cites.append((kw["src"], kw["dst"]))
                    return type("R", (), {"single": lambda self: None})()
                raise AssertionError(f"unexpected cypher: {cypher[:60]}")

        return _Session()


_DOC = {
    "title": "Attention Is All You Need",
    "url": "http://example.com",
    "source": "arxiv",
    "published_date": "2017-06-12",
    "authors": ["Ashish Vaswani", "Noam Shazeer", "  "],  # blank is dropped
    "references": [],
}


def test_authors_stored_as_property_by_default(monkeypatch):
    """Default (flag off): authors land on the Paper node as a property; no
    :AUTHORED edges, no separate Author entity upserts."""
    monkeypatch.setattr("graph.merger._encode",
                        lambda names: np.eye(max(len(names), 1), dtype="float32"))
    monkeypatch.setattr("graph.merger.cfg.EXTRACT_AUTHORS", False, raising=False)

    driver = _PaperRecordingDriver()
    merger = KGMerger(driver)
    merger.upsert_paper(dict(_DOC))

    # Exactly one entity upsert — the Paper itself (no Author entities).
    assert len(driver.entities) == 1
    paper_name, paper_props = driver.entities[0]
    assert paper_name == "Attention Is All You Need"

    # Authors present as a cleaned list property on the Paper node.
    assert paper_props["authors"] == ["Ashish Vaswani", "Noam Shazeer"]

    # No embedded author subgraph.
    assert driver.authored == []


def test_authors_subgraph_restored_when_flag_on(monkeypatch):
    """Flag on: AUTHORED edges are written and authors become entities, but the
    authors property is still recorded on the Paper node too."""
    monkeypatch.setattr("graph.merger._encode",
                        lambda names: np.eye(max(len(names), 1), dtype="float32"))
    monkeypatch.setattr("graph.merger.cfg.EXTRACT_AUTHORS", True, raising=False)

    driver = _PaperRecordingDriver()
    merger = KGMerger(driver)
    merger.upsert_paper(dict(_DOC))

    assert ("Ashish Vaswani", "Attention Is All You Need") in driver.authored
    assert ("Noam Shazeer", "Attention Is All You Need") in driver.authored

    # Author entities get an explicit null name_embedding → stay out of the
    # vector index.
    author_entities = {n: p for (n, p) in driver.entities if n in
                       ("Ashish Vaswani", "Noam Shazeer")}
    assert set(author_entities) == {"Ashish Vaswani", "Noam Shazeer"}
    for props in author_entities.values():
        assert props.get("name_embedding") is None


# ── Extractor-side defensive filter ────────────────────────────────────────────

class _Triple:
    def __init__(self, subject, predicate, obj, subject_type="Concept", obj_type="Concept"):
        self.subject = subject
        self.predicate = predicate
        self.obj = obj
        self.confidence = 1.0
        self.source_id = ""
        self.subject_type = subject_type
        self.obj_type = obj_type
        self.subject_props = {}
        self.obj_props = {}


def _run_filter_with_flag(monkeypatch, flag_value):
    import ingestion.extractors as ex

    triples = [
        _Triple("Transformer", "proposes", "Self-Attention",
                subject_type="Model", obj_type="ArchitecturalInnovation"),
        _Triple("Ashish Vaswani", "developed_by", "Transformer",
                subject_type="Author", obj_type="Model"),
    ]

    class _FakeExtractor:
        last_results = []

        def extract(self, text, source_id="", metadata=None):
            return triples

    monkeypatch.setattr(ex, "get_extractor", lambda: _FakeExtractor())
    monkeypatch.setattr(ex.cfg, "EXTRACT_AUTHORS", flag_value, raising=False)

    kept, _results = ex.extract_triples_and_results("irrelevant text", "doc1")
    return kept


def test_author_triple_dropped_when_flag_off(monkeypatch):
    kept = _run_filter_with_flag(monkeypatch, False)
    kinds = {(t.subject_type, t.obj_type) for t in kept}
    # The Author-typed triple is gone; the Model->Innovation triple survives.
    assert ("Author", "Model") not in kinds
    assert ("Model", "ArchitecturalInnovation") in kinds
    assert len(kept) == 1


def test_author_triple_kept_when_flag_on(monkeypatch):
    kept = _run_filter_with_flag(monkeypatch, True)
    assert len(kept) == 2
