"""
ingestion/extractors.py
Extracts two things from each paper:
  1. (subject) -[predicate]-> (object) triples  — concepts, methods, entities
  2. Paper node with edges to every concept it introduces/uses
     + paper-to-paper relationships (builds_on, cites, prerequisite_for)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import google.generativeai as genai

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    subject: str
    predicate: str
    obj: str
    obj: str
    confidence: float = 1.0
    source_id: str = ""
    subject_type: str = "Unknown"
    obj_type: str = "Unknown"

    def __repr__(self):
        return f"({self.subject}:{self.subject_type}) --[{self.predicate}]--> ({self.obj}:{self.obj_type})"


@dataclass
class PaperNode:
    """Structured representation of a paper and everything it contains."""
    paper_id: str
    title: str
    year: str = ""
    authors: list[str] = field(default_factory=list)
    url: str = ""
    citations: int = 0
    concepts: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    builds_on: list[str] = field(default_factory=list)
    difficulty: str = "intermediate"
    one_line_summary: str = ""
    triples: list[Triple] = field(default_factory=list)


_TRIPLE_PROMPT = """You are a scientific knowledge graph extractor.
Given a research paper abstract, extract factual (subject, predicate, object) triples.

Rules:
- subjects and objects: named entities — methods, models, datasets, concepts, authors, organisations
- predicates: proposes, outperforms, trained_on, published_by, cites, uses_dataset,
  achieves_score_on, related_to, introduces, evaluates, improves, builds_on, uses
- classify every subject and object into ONE type:
    Method        — algorithms, techniques, approaches (RAG, LoRA, attention)
    Model         — specific trained models (GPT-4, BERT, LLaMA)
    Dataset       — datasets and benchmarks (SQuAD, MMLU, ImageNet)
    Author        — person names (Lewis et al., Vaswani)
    Organisation  — companies/labs (OpenAI, Google DeepMind, Meta)
    Metric        — evaluation metrics (BLEU, F1, perplexity, accuracy)
    Concept       — abstract ideas (hallucination, grounding, fine-tuning)
    Paper         — paper titles or references
    Unknown       — if none of the above fit
- return ONLY a valid JSON array, no markdown
- each element: {
    "subject": "...",
    "subject_type": "Method|Model|Dataset|Author|Organisation|Metric|Concept|Paper|Unknown",
    "predicate": "...",
    "object": "...",
    "object_type": "Method|Model|Dataset|Author|Organisation|Metric|Concept|Paper|Unknown",
    "confidence": 0.0-1.0
  }"""


_PAPER_ANALYSIS_PROMPT = """You are a research paper analyst building a knowledge graph.
Given a paper title and abstract, extract structured information.

Return ONLY valid JSON, no markdown:
{
  "concepts": ["key concepts/ideas introduced or central to this paper, max 8"],
  "methods": ["methods/techniques proposed or used, max 6"],
  "datasets": ["datasets used for training or evaluation, max 5"],
  "metrics": ["evaluation metrics used, max 5"],
  "key_findings": ["2-3 main contributions or results as short strings"],
  "builds_on": ["titles of papers this work directly builds upon"],
  "difficulty": "beginner|intermediate|advanced",
  "one_line_summary": "one sentence describing what this paper does and why it matters"
}"""


class LLMExtractor:
    def __init__(self):
        genai.configure(api_key=cfg.GEMINI_API_KEY)
        self._triple_model = genai.GenerativeModel(
            model_name=cfg.LLM_MODEL,
            system_instruction=_TRIPLE_PROMPT,
        )
        self._analysis_model = genai.GenerativeModel(
            model_name=cfg.LLM_MODEL,
            system_instruction=_PAPER_ANALYSIS_PROMPT,
        )

    def extract(self, text: str, source_id: str = "") -> list[Triple]:
        try:
            response = self._triple_model.generate_content(text[:4000])
            raw = response.text.strip()
            raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
            items = json.loads(raw)
            return [
                Triple(
                    subject=item["subject"],
                    predicate=item["predicate"],
                    obj=item["object"],
                    confidence=float(item.get("confidence", 1.0)),
                    source_id=source_id,
                    subject_type=item.get("subject_type", "Unknown"),
                    obj_type=item.get("object_type", "Unknown"),
                    subject_type=item.get("subject_type", "Unknown"),
                    obj_type=item.get("object_type", "Unknown"),
                )
                for item in items
                if all(k in item for k in ("subject", "predicate", "object"))
            ]
        except Exception as exc:
            logger.exception("Triple extraction failed for %s", source_id)
            return []

    def analyse_paper(self, doc: dict) -> PaperNode:
        title   = doc.get("title", "")
        abstract = doc.get("abstract", "")
        text = f"Title: {title}\n\nAbstract: {abstract}"

        # Parse year safely
        raw_date = str(doc.get("published_date", doc.get("published", "")) or "")
        year = raw_date[:4] if raw_date else ""

        paper = PaperNode(
            paper_id=doc.get("id", ""),
            title=title,
            year=year,
            authors=doc.get("authors", []),
            url=doc.get("url", ""),
            citations=int(doc.get("citations") or 0),
        )

        try:
            response = self._analysis_model.generate_content(text[:4000])
            raw = response.text.strip()
            raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(raw)

            paper.concepts      = data.get("concepts", [])[:8]
            paper.methods       = data.get("methods", [])[:6]
            paper.datasets      = data.get("datasets", [])[:5]
            paper.metrics       = data.get("metrics", [])[:5]
            paper.key_findings  = data.get("key_findings", [])[:3]
            paper.builds_on     = data.get("builds_on", [])[:5]
            paper.difficulty    = data.get("difficulty", "intermediate")
            paper.one_line_summary = data.get("one_line_summary", "")

        except Exception as exc:
            logger.warning("Paper analysis failed for '%s': %s", title, exc)

        paper.triples = self.extract(f"{title}. {abstract}", source_id=doc.get("id", ""))
        return paper


class RebelExtractor:
    def __init__(self):
        from transformers import pipeline as hf_pipeline
        self._pipe = hf_pipeline("text2text-generation", model="Babelscape/rebel-large", tokenizer="Babelscape/rebel-large")

    def extract(self, text: str, source_id: str = "") -> list[Triple]:
        try:
            outputs = self._pipe(text[:1024], return_tensors=True, return_text=False, max_length=512)
            decoded = self._pipe.tokenizer.batch_decode([o["generated_token_ids"] for o in outputs], skip_special_tokens=False)
            triples, current = [], {}
            for token in decoded[0].split("<"):
                token = token.strip()
                if token.startswith("triplet>"): 
                    if current: triples.append(current)
                    current = {}
                elif token.startswith("subj>"): current["subject"] = token[5:].strip()
                elif token.startswith("rel>"): current["predicate"] = token[4:].strip()
                elif token.startswith("obj>"): current["obj"] = token[4:].strip()
            if current: triples.append(current)
            return [Triple(subject=t["subject"],predicate=t["predicate"],obj=t["obj"],confidence=0.85,source_id=source_id)
                    for t in triples if all(k in t for k in ("subject","predicate","obj"))]
        except Exception as exc:
            logger.warning("Rebel failed: %s", exc); return []


class GLiNERExtractor:
    def __init__(self):
        from gliner import GLiNER as _GLiNER
        self._model = _GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    def extract(self, text: str, source_id: str = "") -> list[Triple]:
        try:
            sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
            triples, labels = [], ["person","organisation","method","model","dataset","concept","metric"]
            for sent in sentences[:30]:
                ents = [e["text"] for e in self._model.predict_entities(sent, labels, threshold=0.5)]
                for i in range(len(ents)-1):
                    triples.append(Triple(subject=ents[i],predicate="related_to",obj=ents[i+1],confidence=0.7,source_id=source_id))
            return triples
        except Exception as exc:
            logger.warning("GLiNER failed: %s", exc); return []


_extractor_cache: dict = {}

def get_extractor(method: str | None = None):
    method = method or cfg.TRIPLE_EXTRACTION_METHOD
    if method not in _extractor_cache:
        if method == "llm":      _extractor_cache[method] = LLMExtractor()
        elif method == "rebel":  _extractor_cache[method] = RebelExtractor()
        elif method == "gliner": _extractor_cache[method] = GLiNERExtractor()
        else: raise ValueError(f"Unknown extraction method: {method}")
    return _extractor_cache[method]


def extract_triples(text: str, source_id: str = "") -> list[Triple]:
    return get_extractor().extract(text, source_id=source_id)


def analyse_paper(doc: dict) -> PaperNode:
    extractor = get_extractor()
    if hasattr(extractor, "analyse_paper"):
        return extractor.analyse_paper(doc)
    paper = PaperNode(paper_id=doc.get("id",""), title=doc.get("title",""),
                      year=str(doc.get("published_date",""))[:4], authors=doc.get("authors",[]), url=doc.get("url",""))
    paper.triples = extract_triples(f"{doc.get('title','')}. {doc.get('abstract','')}", source_id=doc.get("id",""))
    return paper