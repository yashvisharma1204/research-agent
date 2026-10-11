"""
tests/test_contributions.py

Verifies Part B — reified paper contributions:

  * The LLM payload parser turns a "contributions" array into validated
    Contribution objects (dropping incomplete ones).
  * extract_triples_results_contributions returns a 3-tuple and filters
    contributions whose structural anchors fail the entity quality gate.
  * KGMerger.upsert_contributions writes the reified subgraph using the
    existing :RELATION {type:...} pattern with the new high-signal predicates
    (makes_claim / addresses / via) so the retriever picks them up unchanged.
  * The new predicates are registered as high-signal (outrank related_to).
"""
import numpy as np
import pytest

from graph.merger import KGMerger


# ── Parsing / extraction ───────────────────────────────────────────────────────

def test_parse_contributions_validates_and_filters():
    from ingestion.extractors import _parse_contributions

    raw = [
        {"statement": "Proposes self-attention for sequence modeling",
         "kind": "proposes_method",
         "addresses": "Long-Range Dependency Modeling", "addresses_type": "Task",
         "via": "Self-Attention", "via_type": "ArchitecturalInnovation"},
        {"statement": "", "addresses": "X"},                 # no statement -> dropped
        {"statement": "A claim with no anchors"},            # no addresses/via -> dropped
        "not-a-dict",                                        # ignored
        {"statement": "Fixes vanishing gradients", "kind": "weird_kind",
         "via": "Residual Connection"},                      # kind normalised
    ]
    out = _parse_contributions(raw, "doc-1")
    assert len(out) == 2
    assert out[0].addresses == "Long-Range Dependency Modeling"
    assert out[0].via == "Self-Attention"
    assert out[0].source_id == "doc-1"
    # Unknown kind normalises to the default.
    assert out[1].normalised_kind() == "proposes_method"


def test_contributions_from_text_pulls_array():
    from ingestion.extractors import _contributions_from_text

    payload = (
        '{"triples": [], "results": [], '
        '"contributions": [{"statement": "S", "via": "Mechanism"}]}'
    )
    arr = _contributions_from_text(payload)
    assert len(arr) == 1 and arr[0]["statement"] == "S"
    # Legacy bare array -> no contributions.
    assert _contributions_from_text('[{"subject": "A"}]') == []


def test_extract_3tuple_filters_low_quality_anchor(monkeypatch):
    import ingestion.extractors as ex
    from ingestion.extractors import Contribution

    class _FakeExtractor:
        last_results = []
        last_contributions = [
            # Good anchors -> kept.
            Contribution(statement="Proposes X", addresses="Question Answering",
                         via="Self-Attention", source_id="doc1"),
            # 'via' is 2 chars -> fails is_high_quality_entity -> dropped.
            Contribution(statement="Y", addresses="Summarization", via="QA",
                         source_id="doc1"),
        ]

        def extract(self, text, source_id="", metadata=None):
            return []  # no triples needed for this test

    monkeypatch.setattr(ex, "get_extractor", lambda: _FakeExtractor())
    monkeypatch.setattr(ex.cfg, "EXTRACT_AUTHORS", False, raising=False)

    triples, results, contribs = ex.extract_triples_results_contributions("t", "doc1")
    assert triples == [] and results == []
    assert len(contribs) == 1
    assert contribs[0].addresses == "Question Answering"


def test_extract_triples_and_results_still_2tuple(monkeypatch):
    """Backward-compat: the old 2-tuple API keeps working."""
    import ingestion.extractors as ex

    class _FakeExtractor:
        last_results = []
        last_contributions = []

        def extract(self, text, source_id="", metadata=None):
            return []

    monkeypatch.setattr(ex, "get_extractor", lambda: _FakeExtractor())
    monkeypatch.setattr(ex.cfg, "EXTRACT_AUTHORS", False, raising=False)

    result = ex.extract_triples_and_results("t", "doc1")
    assert isinstance(result, tuple) and len(result) == 2


# ── Merger upsert ──────────────────────────────────────────────────────────────

class _ContribRecordingDriver:
    """Records entity upserts and :RELATION edges keyed by their `type`."""

    def __init__(self):
        self.entities = []            # (name, props)
        self.relations = []           # (subj, type, obj)

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
                if "MERGE (s)-[r:RELATION" in cypher:
                    driver.relations.append((kw["subj"], kw["pred"], kw["obj"]))
                    return type("R", (), {"single": lambda self: None})()
                raise AssertionError(f"unexpected cypher: {cypher[:60]}")

        return _Session()


def test_upsert_contributions_writes_reified_subgraph(monkeypatch):
    from ingestion.extractors import Contribution

    monkeypatch.setattr("graph.merger._encode",
                        lambda names: np.eye(max(len(names), 1), dtype="float32"))
    driver = _ContribRecordingDriver()
    merger = KGMerger(driver)

    contribs = [
        Contribution(
            statement="Proposes self-attention to model long-range dependencies",
            kind="proposes_method",
            addresses="Long-Range Dependency Modeling", addresses_type="Task",
            via="Self-Attention", via_type="ArchitecturalInnovation",
            source_id="1706.03762",
        ),
    ]
    n = merger.upsert_contributions(contribs, "1706.03762",
                                    paper_name="Attention Is All You Need")
    assert n == 1

    types = {t for (_s, t, _o) in driver.relations}
    assert {"makes_claim", "addresses", "via"} <= types

    # makes_claim originates from the PAPER node (title), not the source id.
    makes = [(s, o) for (s, t, o) in driver.relations if t == "makes_claim"]
    assert len(makes) == 1
    paper, contrib_node = makes[0]
    assert paper == "Attention Is All You Need"
    assert contrib_node.startswith("1706.03762::contrib::")

    # addresses -> the Task problem; via -> the mechanism.
    addressed = [o for (s, t, o) in driver.relations if t == "addresses"]
    viad = [o for (s, t, o) in driver.relations if t == "via"]
    assert addressed == ["Long-Range Dependency Modeling"]
    assert viad == ["Self-Attention"]

    # The Contribution node itself is upserted with an explicit null embedding.
    contrib_entities = [p for (nm, p) in driver.entities if nm == contrib_node]
    assert contrib_entities and contrib_entities[0].get("name_embedding") is None
    assert contrib_entities[0]["kind"] == "proposes_method"


def test_upsert_contributions_deterministic_id(monkeypatch):
    """Same statement -> same contrib id across calls (idempotent re-ingest)."""
    from ingestion.extractors import Contribution

    monkeypatch.setattr("graph.merger._encode",
                        lambda names: np.eye(max(len(names), 1), dtype="float32"))

    def run_once():
        driver = _ContribRecordingDriver()
        merger = KGMerger(driver)
        merger.upsert_contributions(
            [Contribution(statement="Proposes X", via="Mechanism", source_id="d")],
            "d", paper_name="P",
        )
        return [o for (s, t, o) in driver.relations if t == "makes_claim"][0]

    assert run_once() == run_once()


def test_upsert_contributions_skips_makes_claim_without_paper(monkeypatch):
    from ingestion.extractors import Contribution

    monkeypatch.setattr("graph.merger._encode",
                        lambda names: np.eye(max(len(names), 1), dtype="float32"))
    driver = _ContribRecordingDriver()
    merger = KGMerger(driver)
    merger.upsert_contributions(
        [Contribution(statement="S", via="Mechanism", source_id="d")],
        "d", paper_name=None,
    )
    types = {t for (_s, t, _o) in driver.relations}
    assert "makes_claim" not in types   # no paper -> no claim edge
    assert "via" in types                # but the mechanism link still exists


# ── Config predicate weights ────────────────────────────────────────────────────

def test_contribution_predicates_are_high_signal():
    from config import get_predicate_weight, HIGH_SIGNAL_PREDICATES

    for pred in ("makes_claim", "addresses", "via"):
        assert pred in HIGH_SIGNAL_PREDICATES
        # Must outrank related_to (0.30) and the unknown default (0.40).
        assert get_predicate_weight(pred) >= 0.90
    assert get_predicate_weight("makes_claim") > get_predicate_weight("related_to")
