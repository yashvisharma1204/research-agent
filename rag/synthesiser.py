"""
rag/synthesiser.py
Takes a RetrievalContext and a question, builds a rich prompt,
calls Gemini, and returns a structured answer with citations.
"""
from __future__ import annotations
from google.api_core import exceptions
import logging
from dataclasses import dataclass, field

import google.generativeai as genai

from config import cfg
from rag.retriever import RetrievalContext

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a precise scientific research assistant backed by a live knowledge graph.

When answering:
1. Use ONLY the graph triples and document excerpts provided in the context.
2. Cite every factual claim with .
3. If the context does not contain enough information to answer, say so clearly.
4. When multiple sources confirm a relationship, note the agreement.
5. Highlight any contradictions found in the context.
6. Keep answers structured: lead with the direct answer, then supporting evidence."""


def _format_triples(triples) -> str:
    if not triples:
        return "No graph triples found."
    lines = []
    for t in triples:
        conf_bar = "★" * min(5, round(t.confidence * 5))
        lines.append(
            f"  ({t.subject}) --[{t.predicate}]--> ({t.obj})"
            f"  [conf={t.confidence:.2f} {conf_bar}, seen {t.mention_count}x]"
        )
    return "\n".join(lines)


def _format_vector(results) -> str:
    if not results:
        return "No document excerpts found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] (score={r.score:.3f}, id={r.doc_id})\n{r.text[:400]}")
    return "\n\n".join(lines)


@dataclass
class Answer:
    question: str
    answer: str
    entity_seeds: list[str] = field(default_factory=list)
    graph_triple_count: int = 0
    vector_result_count: int = 0
    model: str = ""


class Synthesiser:
    def __init__(self):
        genai.configure(api_key=cfg.GEMINI_API_KEY)
        self._model = genai.GenerativeModel(
            model_name=cfg.LLM_MODEL,
            system_instruction=_SYSTEM_PROMPT,
        )

    def answer(self, question: str, context: RetrievalContext) -> Answer:
        try: 
            prompt = f"""## Knowledge graph triples (multi-hop, ordered by confidence)

            {_format_triples(context.graph_triples)}

            ## Relevant document excerpts (semantic search)

            {_format_vector(context.vector_results)}

            ## Seed entities found in your question

            {', '.join(context.entity_names) if context.entity_names else 'none detected'}

            ---

            Question: {question}"""

            response = self._model.generate_content(prompt)
            answer_text = response.text
            logger.info("Synthesised answer (%d chars)", len(answer_text))

            return Answer(
                question=question,
                answer=answer_text,
                entity_seeds=context.entity_names,
                graph_triple_count=len(context.graph_triples),
                vector_result_count=len(context.vector_results),
                model=cfg.LLM_MODEL,
            )
        except exceptions.ResourceExhausted:
            logger.error("Gemini API Quota Exceeded!")
            return Answer(
                question=question,
                answer="⚠️ API Quota Reached. Please wait a minute.",
                entity_seeds=[], graph_triple_count=0, vector_result_count=0, route="ERROR", model="Gemini"
            )
        except exceptions.NotFound:
            logger.error("Gemini Model Not Found! Check model string.")
            return Answer(
                question=question,
                answer="⚠️ Model deprecated. Please ensure the backend is using 'gemini-2.5-flash'.",
                entity_seeds=[], graph_triple_count=0, vector_result_count=0, route="ERROR", model="Gemini"
            )
        except exceptions.GoogleAPICallError as e:
            # This acts as a catch-all for any other Google API failure
            logger.error(f"Google API Error: {e}")
            return Answer(
                question=question,
                answer="⚠️ Google API Connection Error.",
                entity_seeds=[], graph_triple_count=0, vector_result_count=0, route="ERROR", model="Gemini"
            )