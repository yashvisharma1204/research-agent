"""
rag/synthesiser.py
Handles both:
  1. RAG answer generation (existing)
  2. Learning path generation (new)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import google.generativeai as genai

from config import cfg
from rag.retriever import RetrievalContext

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = """You are a precise scientific research assistant backed by a live knowledge graph.

When answering:
1. Use ONLY the graph triples and document excerpts provided in the context.
2. Cite every factual claim with [source: <paper_id or entity>].
3. If the context does not contain enough information, say so clearly.
4. When multiple sources confirm a relationship, note the agreement.
5. Highlight any contradictions found in the context.
6. Keep answers structured: lead with the direct answer, then supporting evidence."""


_LEARNING_PATH_SYSTEM = """You are a research mentor helping someone build deep knowledge in a topic.

Given a list of papers with their metadata, create an ordered learning path.

Rules:
- Order papers from most foundational to most advanced
- Earlier papers should be prerequisites for later ones
- Give a specific, useful reason for each paper (not generic)
- The reason should say what the reader will gain from it
- Group papers into stages: Foundation → Core Methods → Advanced → Current Research

Return ONLY valid JSON, no markdown:
{
  "topic": "...",
  "overview": "2-3 sentences on what this learning path covers and what the reader will be able to do after",
  "stages": [
    {
      "name": "Foundation",
      "description": "...",
      "papers": [
        {
          "title": "...",
          "year": "...",
          "reason": "Read this first because...",
          "key_concepts": ["concept1", "concept2"],
          "difficulty": "beginner|intermediate|advanced",
          "estimated_read_time": "X hours",
          "url": "..."
        }
      ]
    }
  ],
  "what_you_will_learn": ["skill1", "skill2", "skill3"]
}"""


@dataclass
class Answer:
    question: str
    answer: str
    entity_seeds: list[str] = field(default_factory=list)
    graph_triple_count: int = 0
    vector_result_count: int = 0
    model: str = ""


@dataclass
class LearningPathStage:
    name: str
    description: str
    papers: list[dict] = field(default_factory=list)


@dataclass
class LearningPath:
    topic: str
    overview: str
    stages: list[LearningPathStage] = field(default_factory=list)
    what_you_will_learn: list[str] = field(default_factory=list)
    total_papers: int = 0


class Synthesiser:
    def __init__(self):
        genai.configure(api_key=cfg.GEMINI_API_KEY)
        self._answer_model = genai.GenerativeModel(
            model_name=cfg.LLM_MODEL,
            system_instruction=_ANSWER_SYSTEM,
        )
        self._path_model = genai.GenerativeModel(
            model_name=cfg.LLM_MODEL,
            system_instruction=_LEARNING_PATH_SYSTEM,
        )

    # ── RAG answer ────────────────────────────────────────────────────────────

    def answer(self, question: str, context: RetrievalContext) -> Answer:
        triples_text = "\n".join(
            f"  ({t.subject}) --[{t.predicate}]--> ({t.obj}) [conf={t.confidence:.2f}]"
            for t in context.graph_triples
        ) or "No graph triples found."

        vector_text = "\n\n".join(
            f"[{i+1}] (score={r.score:.3f}, id={r.doc_id})\n{r.text[:400]}"
            for i, r in enumerate(context.vector_results)
        ) or "No document excerpts found."

        prompt = f"""## Knowledge graph triples
{triples_text}

## Relevant document excerpts
{vector_text}

## Seed entities: {', '.join(context.entity_names) if context.entity_names else 'none'}

Question: {question}"""

        response = self._answer_model.generate_content(prompt)
        return Answer(
            question=question,
            answer=response.text,
            entity_seeds=context.entity_names,
            graph_triple_count=len(context.graph_triples),
            vector_result_count=len(context.vector_results),
            model=cfg.LLM_MODEL,
        )

    # ── Learning path ─────────────────────────────────────────────────────────

    def generate_learning_path(self, topic: str, papers: list[dict]) -> LearningPath:
        """
        Given a list of paper dicts (from fetchers + analysis),
        generate an ordered learning path with stages.
        """
        import json, re

        # Build paper summary for the prompt
        papers_text = ""
        for i, p in enumerate(papers, 1):
            papers_text += f"""
Paper {i}:
  Title: {p.get('title', '?')}
  Year: {p.get('year', p.get('published_date', '?'))[:4] if p.get('year') or p.get('published_date') else '?'}
  Citations: {p.get('citations', 0) or 0}
  Summary: {p.get('one_line_summary') or p.get('abstract', '')[:200]}
  Concepts: {', '.join(p.get('concepts', [])[:5])}
  Methods: {', '.join(p.get('methods', [])[:4])}
  Difficulty: {p.get('difficulty', 'intermediate')}
  Builds on: {', '.join(p.get('builds_on', [])[:3])}
  URL: {p.get('url', '')}
"""

        prompt = f"Topic: {topic}\n\nPapers to arrange into a learning path:\n{papers_text}"

        try:
            response = self._path_model.generate_content(prompt)
            raw = response.text.strip()
            raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(raw)

            stages = [
                LearningPathStage(
                    name=s.get("name", ""),
                    description=s.get("description", ""),
                    papers=s.get("papers", []),
                )
                for s in data.get("stages", [])
            ]

            total = sum(len(s.papers) for s in stages)

            return LearningPath(
                topic=data.get("topic", topic),
                overview=data.get("overview", ""),
                stages=stages,
                what_you_will_learn=data.get("what_you_will_learn", []),
                total_papers=total,
            )

        except Exception as exc:
            logger.exception("Learning path generation failed for topic '%s'", topic)
            # Fallback — simple ordered list
            return LearningPath(
                topic=topic,
                overview=f"A learning path for {topic}.",
                stages=[LearningPathStage(
                    name="All Papers",
                    description="Papers ordered by year",
                    papers=[{
                        "title": p.get("title",""),
                        "year": str(p.get("year",""))[:4],
                        "reason": p.get("one_line_summary") or "Key paper in this topic",
                        "key_concepts": p.get("concepts",[])[:3],
                        "difficulty": p.get("difficulty","intermediate"),
                        "url": p.get("url",""),
                    } for p in sorted(papers, key=lambda x: str(x.get("year","9999")))]
                )],
                total_papers=len(papers),
            )