"""
ingestion/scheduler.py
Prefect flows that fetch papers on a schedule, extract triples,
and push everything into the knowledge graph.

Run standalone:  python -m ingestion.scheduler
"""
from __future__ import annotations

import logging
from datetime import timedelta

from prefect import flow, task
from prefect.schedules import IntervalSchedule

from config import cfg
from graph.merger import KGMerger
from graph.neo4j_client import get_driver
from ingestion.extractors import extract_triples
from ingestion.fetchers import fetch_arxiv, fetch_rss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(retries=3, retry_delay_seconds=30)
def ingest_arxiv(topic: str) -> list[dict]:
    return fetch_arxiv(topic, max_results=cfg.MAX_PAPERS_PER_TOPIC)


@task(retries=2, retry_delay_seconds=15)
def ingest_rss() -> list[dict]:
    return fetch_rss()


@task
def process_document(doc: dict, merger: KGMerger) -> int:
    """Extract triples from one document and merge into the graph."""
    text = f"{doc['title']}. {doc['abstract']}"
    triples = extract_triples(text, source_id=doc["id"])

    # Also upsert the paper node itself with metadata
    merger.upsert_entity(
        name=doc["title"],
        entity_type="Paper",
        properties={
            "url": doc.get("url", ""),
            "published_date": doc.get("published_date", ""),
            "authors": doc.get("authors", []),
            "source": doc.get("source", ""),
        },
    )

    for triple in triples:
        merger.upsert_triple(
            subj=triple.subject,
            pred=triple.predicate,
            obj=triple.obj,
            source_paper=doc["id"],
            confidence=triple.confidence,
        )

    logger.info("Merged %d triples from '%s'", len(triples), doc["title"][:60])
    return len(triples)


# ── Main flow ─────────────────────────────────────────────────────────────────

@flow(name="research-agent-ingest", log_prints=True)
def daily_ingest_flow(topics: list[str] | None = None):
    topics = topics or cfg.RESEARCH_TOPICS
    driver = get_driver()
    merger = KGMerger(driver)

    all_docs: list[dict] = []

    # arXiv per topic
    for topic in topics:
        docs = ingest_arxiv(topic)
        all_docs.extend(docs)

    # RSS feeds
    rss_docs = ingest_rss()
    all_docs.extend(rss_docs)

    # Deduplicate by ID
    seen: set[str] = set()
    unique_docs = []
    for doc in all_docs:
        if doc["id"] not in seen:
            seen.add(doc["id"])
            unique_docs.append(doc)

    logger.info("Processing %d unique documents", len(unique_docs))
    total_triples = 0
    for doc in unique_docs:
        total_triples += process_document(doc, merger)

    logger.info("Ingest complete. Total triples merged: %d", total_triples)
    driver.close()
    return {"documents": len(unique_docs), "triples": total_triples}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run once immediately, then on schedule
    daily_ingest_flow()

    daily_ingest_flow.serve(
        name="daily-research-ingest",
        schedule=IntervalSchedule(interval=timedelta(hours=cfg.FETCH_INTERVAL_HOURS)),
    )