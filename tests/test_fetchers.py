import ingestion.fetchers as fetchers


def test_fetch_foundational_prefers_title_lookup_for_paper_names(monkeypatch):
    calls = []

    def fake_fetch_by_title(title, max_results=5):
        calls.append(("title", title, max_results))
        return [{"id": "paper-1", "title": title}]

    def fake_fetch_by_citations(topic, max_results=10):
        calls.append(("citations", topic, max_results))
        return [{"id": "paper-2", "title": topic}]

    monkeypatch.setattr(fetchers, "fetch_by_title", fake_fetch_by_title)
    monkeypatch.setattr(fetchers, "fetch_by_citations", fake_fetch_by_citations)

    results = fetchers.fetch_foundational("Attention Is All You Need", max_results=3)

    assert results[0]["title"] == "Attention Is All You Need"
    assert calls[0][0] == "title"


def test_fetch_foundational_uses_citations_for_concept_queries(monkeypatch):
    calls = []

    def fake_fetch_by_title(title, max_results=5):
        calls.append(("title", title, max_results))
        return []

    def fake_fetch_by_citations(topic, max_results=10):
        calls.append(("citations", topic, max_results))
        return [{"id": "paper-3", "title": "Popular concept paper"}]

    monkeypatch.setattr(fetchers, "fetch_by_title", fake_fetch_by_title)
    monkeypatch.setattr(fetchers, "fetch_by_citations", fake_fetch_by_citations)

    results = fetchers.fetch_foundational("retrieval augmented generation", max_results=3)

    assert results[0]["title"] == "Popular concept paper"
    assert calls[0][0] == "title"
    assert calls[1][0] == "citations"
