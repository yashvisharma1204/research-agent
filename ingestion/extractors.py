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
    obj: str          # 'object' is a Python builtin; use obj
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


class RebelExtractor:
    """
    Uses the Babelscape/rebel-large model.
    Install: pip install transformers torch
    """

    def __init__(self):
        from transformers import pipeline as hf_pipeline
        self._pipe = hf_pipeline(
            "text2text-generation",
            model="Babelscape/rebel-large",
            tokenizer="Babelscape/rebel-large",
        )

    @staticmethod
    def _parse_rebel_output(text: str) -> list[dict]:
        triples = []
        current = {}
        for token in text.split("<"):
            token = token.strip()
            if token.startswith("triplet>"):
                if current:
                    triples.append(current)
                current = {}
            elif token.startswith("subj>"):
                current["subject"] = token[5:].strip()
            elif token.startswith("rel>"):
                current["predicate"] = token[4:].strip()
            elif token.startswith("obj>"):
                current["obj"] = token[4:].strip()
        if current:
            triples.append(current)
        return triples

    def extract(self, text: str, source_id: str = "") -> list[Triple]:
        try:
            outputs = self._pipe(
                text[:1024],
                return_tensors=True,
                return_text=False,
                max_length=512,
            )
            decoded = self._pipe.tokenizer.batch_decode(
                [o["generated_token_ids"] for o in outputs],
                skip_special_tokens=False,
            )
            raw_triples = self._parse_rebel_output(decoded[0])
            return [
                Triple(
                    subject=t["subject"],
                    predicate=t["predicate"],
                    obj=t["obj"],
                    confidence=0.85,
                    source_id=source_id,
                )
                for t in raw_triples
                if all(k in t for k in ("subject", "predicate", "obj"))
            ]
        except Exception as exc:
            logger.warning("Rebel extraction failed for %s: %s", source_id, exc)
            return []


_GLINER_LABELS = [
    "person", "organization", "method", "model", "dataset",
    "concept", "disease", "drug", "gene", "metric",
]

_SIMPLE_PREDICATES = [
    ("proposes", ["proposes", "introduces", "presents"]),
    ("outperforms", ["outperforms", "beats", "surpasses", "exceeds"]),
    ("uses", ["uses", "utilises", "employs", "applies"]),
    ("trained_on", ["trained on", "fine-tuned on"]),
    ("related_to", []),   # fallback
]


class GLiNERExtractor:
    def __init__(self):
        from gliner import GLiNER as _GLiNER
        self._model = _GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    def _find_predicate(self, sent: str, s: str, o: str) -> str:
        between = sent[sent.find(s) + len(s): sent.find(o)].lower()
        for pred, keywords in _SIMPLE_PREDICATES[:-1]:
            if any(kw in between for kw in keywords):
                return pred
        return "related_to"

    def extract(self, text: str, source_id: str = "") -> list[Triple]:
        try:
            sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
            triples = []
            for sent in sentences[:30]:
                entities = self._model.predict_entities(sent, _GLINER_LABELS, threshold=0.5)
                ents = [e["text"] for e in entities]
                for i in range(len(ents) - 1):
                    pred = self._find_predicate(sent, ents[i], ents[i + 1])
                    triples.append(Triple(
                        subject=ents[i], predicate=pred, obj=ents[i + 1],
                        confidence=0.7, source_id=source_id,
                    ))
            return triples
        except Exception as exc:
            logger.warning("GLiNER extraction failed for %s: %s", source_id, exc)
            return []


_extractor_cache: dict[str, object] = {}


def get_extractor(method: str | None = None):
    method = method or cfg.TRIPLE_EXTRACTION_METHOD
    if method not in _extractor_cache:
        if method == "llm":
            _extractor_cache[method] = LLMExtractor()
        elif method == "rebel":
            _extractor_cache[method] = RebelExtractor()
        elif method == "gliner":
            _extractor_cache[method] = GLiNERExtractor()
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    return _extractor_cache[method]


def extract_triples(text: str, source_id: str = "") -> list[Triple]:
    """Convenience wrapper — uses whichever backend is configured."""
    return get_extractor().extract(text, source_id=source_id)
