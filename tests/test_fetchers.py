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


def test_fetch_foundational_uses_curated_list_for_known_concepts(monkeypatch):
    """Path A: known concepts (via TOPIC_ALIASES / FOUNDATIONAL_PAPERS) resolve to
    the curated foundational reading list, not a live citation search."""
    calls = []

    def fake_fetch_by_title(title, max_results=5):
        calls.append(("title", title, max_results))
        return []

    def fake_fetch_by_arxiv_ids(paper_ids):
        calls.append(("curated", tuple(paper_ids)))
        return [{"id": pid, "title": f"Curated {pid}"} for pid in paper_ids]

    def fake_fetch_by_citations(topic, max_results=10):
        calls.append(("citations", topic, max_results))
        return [{"id": "paper-3", "title": "Popular concept paper"}]

    monkeypatch.setattr(fetchers, "fetch_by_title", fake_fetch_by_title)
    monkeypatch.setattr(fetchers, "fetch_by_arxiv_ids", fake_fetch_by_arxiv_ids)
    monkeypatch.setattr(fetchers, "fetch_by_citations", fake_fetch_by_citations)

    results = fetchers.fetch_foundational("retrieval augmented generation", max_results=3)

    # Curated list wins for a known concept; citations is never consulted.
    assert results, "expected curated papers to be returned"
    assert all(r["title"].startswith("Curated ") for r in results)
    assert any(c[0] == "curated" for c in calls)
    assert not any(c[0] == "citations" for c in calls)


def test_fetch_foundational_uses_citations_for_unknown_concepts(monkeypatch):
    """An unknown concept (no curated list, not title-like) falls back to
    a live citation search."""
    calls = []

    def fake_fetch_by_title(title, max_results=5):
        calls.append(("title", title, max_results))
        return []

    def fake_fetch_by_citations(topic, max_results=10):
        calls.append(("citations", topic, max_results))
        return [{"id": "paper-3", "title": "Popular concept paper"}]

    monkeypatch.setattr(fetchers, "fetch_by_title", fake_fetch_by_title)
    monkeypatch.setattr(fetchers, "fetch_by_citations", fake_fetch_by_citations)

    # "quantum annealing" is not in FOUNDATIONAL_PAPERS/TOPIC_ALIASES and is not
    # title-like (lowercase, no title keywords).
    results = fetchers.fetch_foundational("quantum annealing", max_results=3)

    assert results[0]["title"] == "Popular concept paper"
    assert any(c[0] == "citations" for c in calls)
