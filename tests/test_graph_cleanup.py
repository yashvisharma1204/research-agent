"""
tests/test_graph_cleanup.py

Covers the graph-quality fixes:

  * URL/DOI strings never become entities — rejected by both the central
    is_high_quality_entity filter and the per-backend _valid_entity_name guard.
  * A :Paper/:Document node is anchored to the entities extracted from its own
    text via (:Paper)-[:MENTIONS]->(:Entity), using the POST-dedup resolved
    names so papers are never orphaned (the "Attention Is All You Need" bug).
"""
import numpy as np

import ingestion.extractors as ex
from graph.merger import KGMerger


# ── URL / DOI entity rejection ──────────────────────────────────────────────

def test_url_like_detection():
    assert ex._is_url_like("http://arxiv.org/abs/1706.03762")
    assert ex._is_url_like("https://example.com/paper")
    assert ex._is_url_like("www.google.com/x")
    assert ex._is_url_like("doi:10.1000/xyz")
    assert ex._is_url_like("arxiv.org/abs/1706.03762")
    # Real entity names must NOT be flagged.
    assert not ex._is_url_like("Transformer")
    assert not ex._is_url_like("Self-Attention")
    assert not ex._is_url_like("GPT-3")


def test_quality_filter_rejects_urls():
    assert not ex.is_high_quality_entity("http://arxiv.org/abs/1706.03762")
    assert not ex.is_high_quality_entity("https://semanticscholar.org/paper/abc")
    # A legitimate entity still passes.
    assert ex.is_high_quality_entity("Transformer")


def test_valid_entity_name_guard():
    assert ex._valid_entity_name("BERT")
    assert not ex._valid_entity_name("")
    assert not ex._valid_entity_name("   ")
    assert not ex._valid_entity_name("http://arxiv.org/abs/1810.04805")


# ── Paper → concept MENTIONS anchoring ──────────────────────────────────────

class _MentionsRecordingDriver:
    """Fake driver recording entity upserts, RELATION writes, and MENTIONS edges.
    All resolution lookups miss so names pass through unchanged."""

    def __init__(self):
        self.entities = []     # names upserted
        self.relations = []    # (subject, predicate, object)
        self.mentions = []     # (paper, [target names])

    def session(self):
        driver = self

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def run(self, cypher, **kw):
                c = cypher.strip()
                if "MERGE (p)-[:MENTIONS]->(e)" in cypher:
                    driver.mentions.append((kw["paper"], list(kw["names"])))
                    return type("R", (), {"single": lambda self: None})()
                if c.startswith("MATCH (e:Entity {name:"):
                    return type("R", (), {"single": lambda self: None})()
                if "db.index.vector.queryNodes" in cypher:
                    return type("R", (), {"single": lambda self: None})()
                if c.startswith("MERGE (e:Entity"):
                    name = kw["name"]
                    driver.entities.append(name)
                    return type("R", (), {"single": lambda self: {"name": name}})()
                if "MERGE (s)-[r:RELATION" in cypher or "[r:RELATION" in cypher:
                    driver.relations.append(
                        (kw.get("subject"), kw.get("predicate"), kw.get("object"))
                    )
                    return type("R", (), {"single": lambda self: None})()
                # Any other structural MERGE/MATCH is a no-op for this test.
                return type("R", (), {"single": lambda self: None})()

        return _Session()


class _T:
    def __init__(self, subject, obj, s_type="Concept", o_type="Concept"):
        self.subject = subject
        self.obj = obj
        self.predicate = "related_to"
        self.confidence = 1.0
        self.source_id = "doc1"
        self.subject_type = s_type
        self.obj_type = o_type
        self.subject_props = {}
        self.obj_props = {}


def test_paper_mentions_links_extracted_entities(monkeypatch):
    # Identity embeddings so no fuzzy merge collapses the distinct names.
    monkeypatch.setattr(
        "graph.merger._encode",
        lambda names: np.eye(max(len(names), 1), dtype="float32"),
    )
    driver = _MentionsRecordingDriver()
    merger = KGMerger(driver)

    triples = [
        _T("Transformer", "Self-Attention", "Model", "ArchitecturalInnovation"),
        _T("Transformer", "Machine Translation", "Model", "Task"),
    ]

    merger.upsert_document_triples(
        triples, "doc1", paper_name="Attention Is All You Need"
    )

    # Exactly one MENTIONS batch, originating from the paper node.
    assert len(driver.mentions) == 1
    paper, targets = driver.mentions[0]
    assert paper == "Attention Is All You Need"
    # Paper is anchored to every distinct extracted entity (and never to itself).
    assert set(targets) == {"Transformer", "Self-Attention", "Machine Translation"}
    assert "Attention Is All You Need" not in targets


def test_no_mentions_without_paper_name(monkeypatch):
    monkeypatch.setattr(
        "graph.merger._encode",
        lambda names: np.eye(max(len(names), 1), dtype="float32"),
    )
    driver = _MentionsRecordingDriver()
    merger = KGMerger(driver)

    merger.upsert_document_triples(
        [_T("Transformer", "Self-Attention")], "doc1"
    )
    # paper_name omitted → no MENTIONS edges written.
    assert driver.mentions == []


# ── Self-reference anaphor rewrite ──────────────────────────────────────────

def test_canonicalise_self_reference_rewrites_anaphors():
    title = "Attention Is All You Need"
    assert ex._canonicalise_self_reference("This paper", title) == title
    assert ex._canonicalise_self_reference("this paper", title) == title
    assert ex._canonicalise_self_reference("This paper.", title) == title  # trailing dot
    assert ex._canonicalise_self_reference("We", title) == title
    assert ex._canonicalise_self_reference("Our method", title) == title
    # Real entities pass through untouched.
    assert ex._canonicalise_self_reference("Transformer", title) == "Transformer"
    # No-op when the title is unknown.
    assert ex._canonicalise_self_reference("This paper", "") == "This paper"


def test_self_reference_triple_survives_filter(monkeypatch):
    """A ('This paper') -> [proposes] -> (Transformer) triple is rewritten to
    the paper title and kept, instead of being dropped as a low-quality entity."""
    class _T2:
        def __init__(self, subject, obj):
            self.subject = subject
            self.obj = obj
            self.predicate = "proposes"
            self.confidence = 1.0
            self.source_id = "doc1"
            self.subject_type = "Paper"
            self.obj_type = "Model"
            self.subject_props = {}
            self.obj_props = {}

    class _FakeExtractor:
        last_results = []
        last_contributions = []

        def extract(self, text, source_id="", metadata=None):
            return [_T2("This paper", "Transformer")]

    monkeypatch.setattr(ex, "get_extractor", lambda: _FakeExtractor())
    monkeypatch.setattr(ex.cfg, "EXTRACT_AUTHORS", False, raising=False)

    triples, _r, _c = ex.extract_triples_results_contributions(
        "irrelevant", "doc1", paper_title="Attention Is All You Need"
    )
    assert len(triples) == 1
    assert triples[0].subject == "Attention Is All You Need"
    assert triples[0].obj == "Transformer"


def test_self_reference_triple_dropped_without_title(monkeypatch):
    """Without a known title the anaphor can't be rewritten, so 'This paper'
    fails the quality filter and the triple is dropped (prior behaviour)."""
    class _T2:
        def __init__(self):
            self.subject = "This paper"
            self.obj = "Transformer"
            self.predicate = "proposes"
            self.confidence = 1.0
            self.source_id = "doc1"
            self.subject_type = "Paper"
            self.obj_type = "Model"
            self.subject_props = {}
            self.obj_props = {}

    class _FakeExtractor:
        last_results = []
        last_contributions = []

        def extract(self, text, source_id="", metadata=None):
            return [_T2()]

    monkeypatch.setattr(ex, "get_extractor", lambda: _FakeExtractor())
    monkeypatch.setattr(ex.cfg, "EXTRACT_AUTHORS", False, raising=False)

    triples, _r, _c = ex.extract_triples_results_contributions(
        "irrelevant", "doc1"
    )
    assert triples == []
