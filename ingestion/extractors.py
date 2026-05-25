"""
ingestion/extractors.py
Three backends for extracting (subject, predicate, object) triples from text.
Select via config.TRIPLE_EXTRACTION_METHOD: "llm" | "rebel" | "gliner"
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

import google.generativeai as genai

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    subject: str
    predicate: str
    obj: str         
    confidence: float = 1.0
    source_id: str = ""

    def __repr__(self):
        return f"({self.subject}) --[{self.predicate}]--> ({self.obj})"


_SYSTEM_PROMPT = """You are a scientific knowledge graph extractor.
Given a text, extract factual (subject, predicate, object) triples.

Rules:
- subjects and objects should be named entities: methods, models, datasets, concepts, authors, organisations
- predicates should be specific verbs: proposes, outperforms, trained_on, published_by, cites, treats, inhibits, evaluates, uses_dataset, achieves_score_on, related_to
- return ONLY a valid JSON array, no markdown, no explanation
- each element: {"subject": "...", "predicate": "...", "object": "...", "confidence": 0.0-1.0}
- confidence reflects how certain the text is (1.0 = stated as fact, 0.6 = implied)"""


class LLMExtractor:
    def __init__(self):
        genai.configure(api_key=cfg.GEMINI_API_KEY)
        self._model = genai.GenerativeModel(
            model_name=cfg.LLM_MODEL,
            system_instruction=_SYSTEM_PROMPT,
        )

    def extract(self, text: str, source_id: str = "") -> list[Triple]:
        try:
            response = self._model.generate_content(text[:4000])
            raw = response.text.strip()
            # Strip accidental markdown fences
            raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
            items = json.loads(raw)
            return [
                Triple(
                    subject=item["subject"],
                    predicate=item["predicate"],
                    obj=item["object"],
                    confidence=float(item.get("confidence", 1.0)),
                    source_id=source_id,
                )
                for item in items
                if all(k in item for k in ("subject", "predicate", "object"))
            ]
        except (json.JSONDecodeError, KeyError, Exception) as exc:
            logger.warning("LLM extraction failed for %s: %s", source_id, exc)
            return []
