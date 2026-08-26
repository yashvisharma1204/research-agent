"""
rag/synthesiser.py
Layer 4 Upgraded: Synthesises answers with a built-in Self-Correction & Faithfulness Audit Loop.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
import json

from google.genai import types
from google.genai import errors as genai_errors

from config import cfg
from rag.retriever import RetrievalContext
from utils.gemini import get_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a precise scientific research assistant backed by a live knowledge graph.

When answering:
1. Use ONLY the graph triples and document excerpts provided in the context.
2. Cite every factual claim using the source entities or doc IDs.
3. If the context does not contain enough information to answer, say so clearly.
4. When multiple sources confirm a relationship, note the agreement.
5. Highlight any contradictions found in the context.
6. Keep answers structured: lead with the direct answer, then supporting evidence."""


_AUDIT_SYSTEM_PROMPT = """You are an independent scientific auditor and fact-checker for a Knowledge Graph RAG system.
Your job is to audit a generated answer against the provided retrieval context (Graph Triples and Document Excerpts).

Check for:
1. Hallucinations: Are there any claims, models, metrics, or relationships stated in the answer that are NOT supported by the context?
2. Distortion: Were entities or relationships misrepresented?

You must return a valid JSON object with EXACTLY these three keys:
{
  "is_faithful": true or false,
  "critique": "A brief explanation of any hallucinations or unsupported claims found (or 'None' if clean)",
  "corrected_answer": "If is_faithful is false, provide a fully corrected version of the answer removing or fixing ungrounded claims. If true, leave this as an empty string."
}
Return ONLY valid JSON. No markdown code blocks, no preamble.
"""


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
    is_faithful: bool = True
    critique_notes: str = "None"


class Synthesiser:
    def __init__(self):
        self._client = get_client()

    def answer(self, question: str, context: RetrievalContext) -> Answer:
        try: 
            prompt = f"""## Knowledge graph triples (multi-hop, ordered by confidence)

            {_format_triples(context.graph_triples)}

            ## Relevant document excerpts (semantic search)

            {_format_vector(context.vector_results)}

            ## Source abstracts for entities on the retrieved paths

            {chr(10).join(context.paper_abstracts) if context.paper_abstracts else 'No source abstracts found.'}

            ## Seed entities found in your question

            {', '.join(context.entity_names) if context.entity_names else 'none detected'}

            ---

            Question: {question}"""

            # 1. Initial Generation Pass
            response = self._client.models.generate_content(
                model=cfg.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    max_output_tokens=2048,
                ),
            )
            answer_text = (response.text or "").strip()
            logger.info("Initial synthesis complete (%d chars)", len(answer_text))

            # 2. Layer 4 Self-Correction Audit Pass
            audit_prompt = f"""
            Context Provided:
            - Triples: {_format_triples(context.graph_triples)}
            - Excerpts: {_format_vector(context.vector_results)}

            User Question: {question}

            Generated Answer to Audit:
            {answer_text}
            """

            is_faithful = True
            critique_notes = "None"

            try:
                audit_res = self._client.models.generate_content(
                    model=cfg.LLM_MODEL,
                    contents=audit_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_AUDIT_SYSTEM_PROMPT,
                        max_output_tokens=512,
                        response_mime_type="application/json",
                    ),
                )
                audit_data = json.loads((audit_res.text or "").strip())
                is_faithful = audit_data.get("is_faithful", True)
                critique_notes = audit_data.get("critique", "None")

                if not is_faithful and audit_data.get("corrected_answer"):
                    logger.warning("Layer 4 Self-Correction Triggered! Fixing ungrounded claims.")
                    answer_text = audit_data["corrected_answer"]
            except Exception as audit_exc:
                logger.warning("Layer 4 Audit pass encountered an error (proceeding with initial answer): %s", audit_exc)

            return Answer(
                question=question,
                answer=answer_text,
                entity_seeds=context.entity_names,
                graph_triple_count=len(context.graph_triples),
                vector_result_count=len(context.vector_results),
                model=cfg.LLM_MODEL,
                is_faithful=is_faithful,
                critique_notes=critique_notes,
            )

        except genai_errors.APIError as e:
            code = getattr(e, "code", None)
            if code == 429:
                logger.error("Gemini API Quota Exceeded!")
                return Answer(
                    question=question,
                    answer="⚠️ API Quota Reached. Please wait a minute.",
                    entity_seeds=[], graph_triple_count=0, vector_result_count=0,
                    model="Gemini", is_faithful=False, critique_notes="Rate Limit",
                )
            if code == 404:
                logger.error("Gemini Model Not Found! Check model string.")
                return Answer(
                    question=question,
                    answer="⚠️ Model deprecated. Please ensure the backend is using 'gemini-2.5-flash'.",
                    entity_seeds=[], graph_triple_count=0, vector_result_count=0,
                    model="Gemini", is_faithful=False, critique_notes="Model Not Found",
                )
            logger.error("Google API Error: %s", e)
            return Answer(
                question=question,
                answer="⚠️ Google API Connection Error.",
                entity_seeds=[], graph_triple_count=0, vector_result_count=0,
                model="Gemini", is_faithful=False, critique_notes=str(e),
            )