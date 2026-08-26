import numpy as np

from graph.merger import KGMerger


def test_resolve_entity_falls_back_when_vector_query_fails(monkeypatch):
    driver = type("Driver", (), {})()
    session = type("Session", (), {})()

    exact_result = type("Result", (), {"single": lambda self: None})()
    vector_result = type("Result", (), {"single": lambda self: (_ for _ in ()).throw(RuntimeError("vector index unavailable"))})()

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, *args, **kwargs):
            if args[0].startswith("MATCH (e:Entity"):
                return exact_result
            raise RuntimeError("unexpected cypher")

    class FakeDriver:
        def session(self):
            return FakeSession()

    merger = KGMerger(FakeDriver())
    monkeypatch.setattr("graph.merger._encode", lambda name: [np.array([0.1])])

    entity_name = merger._resolve_entity("Alpha", "Person")

    assert entity_name == "Alpha"


class _Triple:
    """Minimal stand-in for ingestion.extractors.Triple."""

    def __init__(self, subject, predicate, obj, subject_type="Concept", obj_type="Concept"):
        self.subject = subject
        self.predicate = predicate
        self.obj = obj
        self.confidence = 1.0
        self.subject_type = subject_type
        self.obj_type = obj_type
        self.subject_props = {}
        self.obj_props = {}


class _RecordingDriver:
    """Fake driver that records upserted entity names and written triples.

    Every entity resolution query returns no exact match and no vector match,
    so resolution falls back to the passed-in name — isolating the in-memory
    dedup logic under test.
    """

    def __init__(self):
        self.upserted_names = []
        self.triples = []
        self.results = []      # (model, dataset, metric, value, rid)
        self.authored = []     # (author, paper)
        self.cites = []        # (src, dst)

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
                    # exact-match lookup -> miss
                    return type("R", (), {"single": lambda self: None})()
                if "db.index.vector.queryNodes" in cypher:
                    # fuzzy vector lookup against existing graph -> miss
                    return type("R", (), {"single": lambda self: None})()
                if c.startswith("MERGE (e:Entity"):
                    driver.upserted_names.append(kw["name"])
                    name = kw["name"]
                    return type("R", (), {"single": lambda self: {"name": name}})()
                if "MERGE (s)-[r:RELATION" in cypher:
                    driver.triples.append((kw["subj"], kw["pred"], kw["obj"]))
                    return type("R", (), {"single": lambda self: None})()
                if "MERGE (res:Result" in cypher:
                    driver.results.append(
                        (kw["model"], kw["dataset"], kw["metric"], kw["value"], kw["rid"])
                    )
                    return type("R", (), {"single": lambda self: None})()
                if "MERGE (a)-[:AUTHORED]->(p)" in cypher:
                    driver.authored.append((kw["author"], kw["paper"]))
                    return type("R", (), {"single": lambda self: None})()
                if "MERGE (src)-[:CITES]->(dst)" in cypher:
                    driver.cites.append((kw["src"], kw["dst"]))
                    return type("R", (), {"single": lambda self: None})()
                raise AssertionError(f"unexpected cypher: {cypher[:60]}")

        return _Session()


def test_batched_exact_dedup_encodes_uniques_once(monkeypatch):
    """A concept repeated across triples is encoded/upserted once (exact dedup)."""
    encode_calls = []

    def fake_encode(names):
        encode_calls.append(list(names))
        return np.eye(len(names), dtype="float32")  # orthogonal -> no fuzzy merge

    monkeypatch.setattr("graph.merger._encode", fake_encode)
    driver = _RecordingDriver()
    merger = KGMerger(driver)

    triples = [
        _Triple("BERT", "uses", "Attention"),
        _Triple("BERT", "improves", "GLUE"),  # BERT repeats
    ]
    written = merger.upsert_document_triples(triples, "doc1")

    assert written == 2
    # One batched encode call covering the 3 unique names.
    assert len(encode_calls) == 1
    assert sorted(encode_calls[0]) == ["Attention", "BERT", "GLUE"]
    # BERT upserted once despite appearing in two triples.
    assert driver.upserted_names.count("BERT") == 1
    assert sorted(driver.upserted_names) == ["Attention", "BERT", "GLUE"]
    assert len(driver.triples) == 2


def test_batched_fuzzy_dedup_collapses_new_near_duplicates(monkeypatch):
    """Two brand-new near-duplicate names in one doc collapse to one node."""

    # Give the two LLM variants near-identical vectors (cosine ~1.0) and a
    # distinct vector for the object.
    vecs = {
        "Large Language Models": np.array([1.0, 0.0], dtype="float32"),
        "large language model": np.array([0.999, 0.0447], dtype="float32"),
        "Reasoning": np.array([0.0, 1.0], dtype="float32"),
    }

    def fake_encode(names):
        return np.stack([vecs[n] for n in names])

    monkeypatch.setattr("graph.merger._encode", fake_encode)
    driver = _RecordingDriver()
    merger = KGMerger(driver)

    triples = [
        _Triple("Large Language Models", "enable", "Reasoning"),
        _Triple("large language model", "enable", "Reasoning"),
    ]
    written = merger.upsert_document_triples(triples, "doc2")

    assert written == 2
    # The two LLM variants collapsed: only the first-seen canonical is upserted.
    assert "Large Language Models" in driver.upserted_names
    assert "large language model" not in driver.upserted_names
    # Both triples reference the canonical subject.
    subjects = {s for (s, _, _) in driver.triples}
    assert subjects == {"Large Language Models"}


# ── Stage A: type -> label whitelist ──────────────────────────────────────────

def test_type_to_label_whitelist():
    from graph.merger import _type_to_label

    assert _type_to_label("Model") == "Model"
    assert _type_to_label("Author") == "Author"
    assert _type_to_label("Paper") == "Paper"
    # Non-whitelisted / fallback types produce no label.
    assert _type_to_label("Unknown") is None
    assert _type_to_label("NotARealType") is None
    assert _type_to_label("") is None
    assert _type_to_label(None) is None


# ── Stage B: results reification ───────────────────────────────────────────────

def test_parse_results_filters_incomplete_and_validates():
    from ingestion.extractors import _parse_results

    raw = [
        {"model": "BERT", "dataset": "SQuAD", "metric": "F1", "value": "88.5"},
        {"model": "BERT", "dataset": "", "metric": "F1", "value": "x"},  # missing dataset
        {"model": "GPT", "metric": "Accuracy", "dataset": "MMLU"},        # no value ok
        "not-a-dict",                                                     # ignored
    ]
    results = _parse_results(raw, "doc-1")
    assert len(results) == 2
    assert results[0].model == "BERT" and results[0].value == "88.5"
    assert results[1].model == "GPT" and results[1].source_id == "doc-1"


def test_safe_parse_payload_object_and_legacy_array():
    from ingestion.extractors import _safe_parse_payload

    # New object form
    triples, results = _safe_parse_payload(
        '{"triples": [{"subject": "A"}], "results": [{"model": "M"}]}'
    )
    assert len(triples) == 1 and len(results) == 1
    # Legacy bare-array form -> treated as triples, empty results
    triples, results = _safe_parse_payload('[{"subject": "A"}]')
    assert len(triples) == 1 and results == []


def test_upsert_results_writes_reified_subgraph(monkeypatch):
    from ingestion.extractors import ScoreResult

    monkeypatch.setattr("graph.merger._encode", lambda names: np.eye(max(len(names), 1), dtype="float32"))
    driver = _RecordingDriver()
    merger = KGMerger(driver)

    results = [
        ScoreResult(model="BERT", dataset="SQuAD", metric="F1", value="88.5", source_id="doc1"),
        ScoreResult(model="BERT", dataset="", metric="F1"),  # invalid -> skipped
    ]
    written = merger.upsert_results(results, "doc1")

    assert written == 1
    # Model, Dataset, Metric all upserted as entities.
    assert "BERT" in driver.upserted_names
    assert "SQuAD" in driver.upserted_names
    assert "F1" in driver.upserted_names
    # One reified Result row with a deterministic id.
    assert len(driver.results) == 1
    model, dataset, metric, value, rid = driver.results[0]
    assert (model, dataset, metric, value) == ("BERT", "SQuAD", "F1", "88.5")
    assert rid == "BERT::SQuAD::F1"


# ── Stage C: author / citation subgraph ────────────────────────────────────────

def test_upsert_authors_creates_authored_edges(monkeypatch):
    monkeypatch.setattr("graph.merger._encode", lambda names: np.eye(max(len(names), 1), dtype="float32"))
    driver = _RecordingDriver()
    merger = KGMerger(driver)

    n = merger.upsert_authors("Attention Is All You Need", ["Ashish Vaswani", "Noam Shazeer", ""])
    assert n == 2
    assert ("Ashish Vaswani", "Attention Is All You Need") in driver.authored
    assert ("Noam Shazeer", "Attention Is All You Need") in driver.authored


def test_upsert_citations_creates_cites_edges_and_skips_self(monkeypatch):
    monkeypatch.setattr("graph.merger._encode", lambda names: np.eye(max(len(names), 1), dtype="float32"))
    driver = _RecordingDriver()
    merger = KGMerger(driver)

    n = merger.upsert_citations("Paper A", ["Paper B", "Paper A", ""])  # self + empty skipped
    assert n == 1
    assert driver.cites == [("Paper A", "Paper B")]
