"""
ingestion/fetchers.py
Pulls raw documents from arXiv, RSS/web feeds, and local PDFs.
Each fetcher returns a list of dicts with a common schema:{id, title, abstract, authors, source, url, published_date}
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import arxiv
import feedparser
import httpx
# import fitz  

logger = logging.getLogger(__name__)


# ── Shared document schema ─────────────────────────────────────────────────────

def _doc(
    doc_id: str,
    title: str,
    abstract: str,
    authors: list[str],
    source: str,
    url: str = "",
    published_date: str = "",
) -> dict:
    return {
        "id": doc_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "source": source,
        "url": url,
        "published_date": published_date,
    }


# ── arXiv ──────────────────────────────────────────────────────────────────────

def fetch_arxiv(query: str, max_results: int = 20) -> list[dict]:
    """Fetch recent papers from arXiv matching *query*."""
    logger.info("Fetching arXiv: %s (max %d)", query, max_results)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client()
    results = []
    for r in client.results(search):
        results.append(
            _doc(
                doc_id=r.entry_id,
                title=r.title,
                abstract=r.summary,
                authors=[a.name for a in r.authors],
                source="arxiv",
                url=r.entry_id,
                published_date=r.published.isoformat() if r.published else "",
            )
        )
    logger.info("arXiv returned %d papers", len(results))
    return results


# ── RSS / Web feeds ────────────────────────────────────────────────────────────

DEFAULT_RSS_FEEDS = [
    "https://rss.arxiv.org/rss/cs.AI",
    "https://rss.arxiv.org/rss/cs.LG",
    "https://rss.arxiv.org/rss/cs.CL",
]


def fetch_rss(feed_urls: list[str] | None = None, max_per_feed: int = 10) -> list[dict]:
    """Fetch entries from RSS feeds."""
    feed_urls = feed_urls or DEFAULT_RSS_FEEDS
    results = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                doc_id = hashlib.md5(entry.get("link", entry.title).encode()).hexdigest()
                results.append(
                    _doc(
                        doc_id=doc_id,
                        title=entry.get("title", ""),
                        abstract=entry.get("summary", ""),
                        authors=[],
                        source="rss",
                        url=entry.get("link", ""),
                        published_date=entry.get("published", ""),
                    )
                )
        except Exception as exc:
            logger.warning("RSS fetch failed for %s: %s", url, exc)
    logger.info("RSS fetched %d entries", len(results))
    return results


def fetch_pdfs(folder: str | Path) -> list[dict]:
    """Extract text from all PDFs in *folder* and return as documents."""
    folder = Path(folder)
    results = []
    for pdf_path in folder.glob("**/*.pdf"):
        try:
            doc = fitz.open(str(pdf_path))
            full_text = "\n".join(page.get_text() for page in doc)
            # Use first 2000 chars as "abstract" for extraction
            abstract = full_text[:2000].strip()
            doc_id = hashlib.md5(str(pdf_path).encode()).hexdigest()
            results.append(
                _doc(
                    doc_id=doc_id,
                    title=pdf_path.stem,
                    abstract=abstract,
                    authors=[],
                    source="pdf",
                    url=str(pdf_path),
                    published_date=datetime.now(timezone.utc).isoformat(),
                )
            )
        except Exception as exc:
            logger.warning("PDF parse failed for %s: %s", pdf_path, exc)
    logger.info("PDF fetcher found %d documents", len(results))
    return results
