"""
ingestion/fetchers.py
Pulls research papers from arXiv, Semantic Scholar, RSS feeds, and local PDFs.

Fetch strategy:
  1. Curated list  — known seminal paper IDs for popular topics
  2. Semantic Scholar — citation-sorted search for any topic (foundational first)
  3. arXiv relevance  — fallback
  4. RSS / PDF        — for live feeds and local documents
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
import time
import arxiv
import feedparser
import httpx

logger = logging.getLogger(__name__)


# ── Shared document schema ────────────────────────────────────────────────────

def _doc(
    doc_id: str,
    title: str,
    abstract: str,
    authors: list[str],
    source: str,
    url: str = "",
    published_date: str = "",
    citations: int | None = None,
) -> dict:
    return {
        "id": doc_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "source": source,
        "url": url,
        "published_date": published_date,
        "citations": citations,
    }


# ── Curated foundational papers by topic (arXiv IDs) ─────────────────────────

FOUNDATIONAL_PAPERS: dict[str, list[str]] = {
    "rag": [
        "2005.11401",   # Original RAG — Lewis et al. 2020
        "2312.10997",   # RAG Survey 2023
        "2301.12652",   # REPLUG
        "2208.09257",   # Atlas
        "2302.00083",   # Self-RAG
        "2305.14283",   # FLARE
    ],
    "llm": [
        "2303.08774",   # GPT-4 technical report
        "2302.13971",   # LLaMA
        "2307.09288",   # LLaMA 2
        "1706.03762",   # Attention is All You Need
        "1810.04805",   # BERT
        "2005.14165",   # GPT-3
    ],
    "transformer": [
        "1706.03762",   # Attention is All You Need
        "1810.04805",   # BERT
        "2005.14165",   # GPT-3
        "2010.11929",   # Vision Transformer (ViT)
        "1910.10683",   # T5
    ],
    "knowledge graph": [
        "2306.08302",   # KG + LLM survey
        "2304.11116",   # Think on Graphs
        "2305.04757",   # GraphRAG
        "2308.06374",   # KnowledGPT
    ],
    "diffusion": [
        "2006.11239",   # DDPM — Ho et al.
        "2010.02502",   # Score-based generative models
        "2112.10752",   # Latent Diffusion / Stable Diffusion
        "2204.05862",   # DALL-E 2
    ],
    "reinforcement learning": [
        "1509.02971",   # DDPG
        "1707.06347",   # PPO
        "1312.5602",    # DQN — Mnih et al.
        "2005.12729",   # Decision Transformer
    ],
    "mixture of experts": [
        "2101.03961",   # Switch Transformer
        "2112.06905",   # GLaM
        "2401.04088",   # Mixtral
    ],
    "cnn": [
        "1409.1556",    # VGG
        "1512.03385",   # ResNet
        "1608.06993",   # DenseNet
        "1409.4842",    # GoogLeNet / Inception
    ],
    "rnn": [
        "1409.3215",    # Seq2Seq — Sutskever et al.
        "1508.04025",   # Attention + RNN — Bahdanau
        "1506.02078",   # LSTM language model
    ],
    "agent": [
        "2210.03629",   # ReAct
        "2303.11366",   # Reflexion
        "2308.11432",   # AutoGen
        "2304.03442",   # Generative Agents
    ],
    "fine tuning": [
        "2106.09685",   # LoRA
        "2110.07602",   # Prefix Tuning
        "2104.08691",   # Prompt Tuning
        "2312.12148",   # QLoRA
    ],
    "multimodal": [
        "2204.05862",   # DALL-E 2
        "2304.08485",   # LLaVA
        "2301.13688",   # BLIP-2
        "2309.17421",   # LLaVA 1.5
    ],
}


# ── Fetch by exact arXiv IDs ──────────────────────────────────────────────────

def fetch_by_arxiv_ids(paper_ids: list[str]) -> list[dict]:
    """Fetch specific papers by their arXiv IDs."""
    logger.info("Fetching %d curated papers by arXiv ID", len(paper_ids))
    client = arxiv.Client()
    search = arxiv.Search(id_list=paper_ids)
    results = []
    for r in client.results(search):
        results.append(_doc(
            doc_id=r.entry_id,
            title=r.title,
            abstract=r.summary,
            authors=[a.name for a in r.authors],
            source="arxiv_curated",
            url=r.entry_id,
            published_date=r.published.isoformat() if r.published else "",
        ))
    return results


# ── Fetch by citation count via Semantic Scholar ──────────────────────────────

def fetch_by_citations(topic: str, max_results: int = 10) -> list[dict]:
    """
    Fetch most-cited papers on any topic using Semantic Scholar API.
    No API key needed. Returns foundational papers first.
    """
    logger.info("Fetching top cited papers for '%s' via Semantic Scholar", topic)
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": topic,
        "limit": max_results,
        "fields": "title,abstract,authors,year,citationCount,externalIds",
    }
    try:
        with httpx.Client(timeout=15) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        logger.warning("Semantic Scholar failed: %s — falling back to arXiv", exc)
        time.sleep(5)  # brief pause before fallback
        return fetch_arxiv(topic, max_results)

    results = []
    for paper in data.get("data", []):
        if not paper.get("abstract"):
            continue
        results.append(_doc(
            doc_id=paper.get("paperId", ""),
            title=paper["title"],
            abstract=paper["abstract"],
            authors=[a["name"] for a in paper.get("authors", [])],
            source="semantic_scholar",
            url=f"https://semanticscholar.org/paper/{paper.get('paperId', '')}",
            published_date=str(paper.get("year", "")),
            citations=paper.get("citationCount", 0),
        ))

    results.sort(key=lambda x: x.get("citations") or 0, reverse=True)
    logger.info(
        "Semantic Scholar: %d papers for '%s', top citations: %s",
        len(results), topic,
        results[0]["citations"] if results else 0,
    )
    return results


# ── Fetch by exact arXiv IDs ──────────────────────────────────────────────────

def fetch_arxiv(paper_ids: list[str]) -> list[dict]:
    """Fetch specific papers by their arXiv IDs with built-in rate-limit handling."""
    logger.info("Fetching %d curated papers by arXiv ID", len(paper_ids))
    
    # Configure the client to space out page chunks and handle retries natively
    client = arxiv.Client(page_size=min(len(paper_ids), 10), delay_seconds=4)
    search = arxiv.Search(id_list=paper_ids)
    results = []
    
    # Implement active backoff loops matching your robust fallback architecture
    for attempt in range(3):
        try:
            for r in client.results(search):
                results.append(_doc(
                    doc_id=r.entry_id,
                    title=r.title,
                    abstract=r.summary,
                    authors=[a.name for a in r.authors],
                    source="arxiv_curated",
                    url=r.entry_id,
                    published_date=r.published.isoformat() if r.published else "",
                ))
            break  # Break loop on successful resolution
        except arxiv.HTTPError as e:
            if e.status == 429:
                wait = 12 * (attempt + 1)
                logger.warning("arXiv curated fetch hit 429 — backing off for %ds (attempt %d/3)", wait, attempt + 1)
                time.sleep(wait)
            else:
                raise
                
    return results

# ── RSS / Web feeds ───────────────────────────────────────────────────────────

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
                doc_id = hashlib.md5(
                    entry.get("link", entry.title).encode()
                ).hexdigest()
                results.append(_doc(
                    doc_id=doc_id,
                    title=entry.get("title", ""),
                    abstract=entry.get("summary", ""),
                    authors=[],
                    source="rss",
                    url=entry.get("link", ""),
                    published_date=entry.get("published", ""),
                ))
        except Exception as exc:
            logger.warning("RSS fetch failed for %s: %s", url, exc)
    logger.info("RSS fetched %d entries", len(results))
    return results


# ── Local PDF ingestion ───────────────────────────────────────────────────────

def fetch_pdfs(folder: str | Path) -> list[dict]:
    """Extract text from all PDFs in *folder* and return as documents."""
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not installed — skipping PDF ingestion")
        return []

    folder = Path(folder)
    results = []
    for pdf_path in folder.glob("**/*.pdf"):
        try:
            doc = fitz.open(str(pdf_path))
            full_text = "\n".join(page.get_text() for page in doc)
            abstract = full_text[:2000].strip()
            doc_id = hashlib.md5(str(pdf_path).encode()).hexdigest()
            results.append(_doc(
                doc_id=doc_id,
                title=pdf_path.stem,
                abstract=abstract,
                authors=[],
                source="pdf",
                url=str(pdf_path),
                published_date=datetime.now(timezone.utc).isoformat(),
            ))
        except Exception as exc:
            logger.warning("PDF parse failed for %s: %s", pdf_path, exc)
    logger.info("PDF fetcher found %d documents", len(results))
    return results


# ── Main entry point ──────────────────────────────────────────────────────────
_cache: dict[str, list[dict]] = {}
def fetch_foundational(topic: str, max_results: int = 10) -> list[dict]:
    cache_key = f"{topic.lower().strip()}:{max_results}"
    if cache_key in _cache:
        logger.info("Cache hit for '%s'", topic)
        return _cache[cache_key]

    topic_clean = topic.lower().strip()
    results: list[dict] = []

    for key in FOUNDATIONAL_PAPERS:
        if key in topic_clean or topic_clean in key:
            logger.info("Using curated paper list for: '%s'", topic_clean)
            results = fetch_by_arxiv_ids(FOUNDATIONAL_PAPERS[key])
            break

    if not results:
        results = fetch_by_citations(topic_clean, max_results)

    if not results:
        results = fetch_arxiv(topic_clean, max_results)

    _cache[cache_key] = results  # always written
    return results
