"""
rag/router.py
Intent classifier that decides whether a query should be answered via:
  - "cypher"  : structured graph traversal (entity lookups, relationship paths)
  - "vector"  : semantic similarity search (open-ended, conceptual questions)
  - "hybrid"  : both (default for ambiguous queries)

Uses Gemini for intent detection with a cheap, low-token prompt.
Falls back to "hybrid" on any error so the system always returns an answer.
"""
from __future__ import annotations

import logging

import google.generativeai as genai

from config import cfg

logger = logging.getLogger(__name__)

RouteType = str   # "cypher" | "vector" | "hybrid"

_ROUTER_SYSTEM = """You are a query router for a research knowledge graph.
Classify the user query into exactly ONE of these route types:

- cypher   : asks about specific named entities, relationships between them,
             or structured facts (e.g. "What does GPT-4 outperform?",
             "Which papers cite BERT?", "How is RAG related to LLMs?")
- vector   : open-ended, conceptual, or thematic questions that need
             semantic similarity over document text
             (e.g. "What are recent trends in NLP?",
             "Summarise approaches to few-shot learning")
- hybrid   : needs both graph structure AND broad semantic context

Reply with ONLY one word: cypher, vector, or hybrid."""


class QueryRouter:
    def __init__(self):
        genai.configure(api_key=cfg.GEMINI_API_KEY)
        self._model = genai.GenerativeModel(
            model_name=cfg.LLM_MODEL,
            system_instruction=_ROUTER_SYSTEM,
        )

    def route(self, query: str) -> RouteType:
        """Return the route type for *query*."""
        try:
            response = self._model.generate_content(query[:500])
            route = response.text.strip().lower()
            if route in ("cypher", "vector", "hybrid"):
                logger.debug("Router: '%s' → %s", query[:60], route)
                return route
            logger.warning("Router returned unexpected value '%s', defaulting to hybrid", route)
            return "hybrid"
        except Exception as exc:
            logger.warning("Router failed (%s), defaulting to hybrid", exc)
            return "hybrid"


# Module-level singleton — imported by retriever.py
_router: QueryRouter | None = None


def get_router() -> QueryRouter:
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router


def route_query(query: str) -> RouteType:
    """Convenience wrapper."""
    return get_router().route(query)
