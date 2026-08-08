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
