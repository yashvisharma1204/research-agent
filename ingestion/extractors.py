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
    subject_type: str = "unknown"
    obj_type: str = "unknown"
    subject_props: dict = None
    obj_props: dict = None

    def __post_init__(self):
        self.subject_props = self.subject_props or {}
        self.obj_props = self.obj_props or {}

    def __repr__(self):
        return f"({self.subject}:{self.subject_type}) --[{self.predicate}]--> ({self.obj}:{self.obj_type})"


_SYSTEM_PROMPT = """You are an advanced scientific knowledge graph extractor.
Given a text, extract factual (subject, predicate, object) triples and rich property payloads.

Rules:
- Subjects and objects should be granular technical entities or mechanism components.
- Predicates should be high-signal semantic verbs: replaces_component_with, introduces_training_objective, ameliorates_limitation_of, benchmarked_superior_to, proposes, uses_dataset, achieves_score_on, related_to.
- Classify every subject and object into ONE of these granular architectural types:
    ArchitecturalInnovation — structural design changes (e.g., Rotary Embedding, SwiGLU, Multi-Query Attention)
    TrainingObjective       — loss formulations, pre-training tasks (e.g., Masked Language Modeling, DPO, RLHF)
    LossFunction            — specific mathematical losses (e.g., Cross-Entropy, Contrastive Loss)
    HardwareConstraint      — compute/memory constraints (e.g., Tensor Parallelism, FlashAttention-2)
    Dataset                 — benchmarks and corpora (e.g., SQuAD, MMLU, CommonCrawl)
    Metric                  — evaluation metrics (e.g., BLEU, Perplexity, Accuracy)
    Concept                 — foundational abstract ideas (e.g., Hallucination, Weight Quantization)
    Paper                   — publication references
    Unknown                 — fallback type
- Extract a "properties" dictionary for each node containing technical attributes (e.g., parameters, dimensions, brief operational summaries if mentioned).
- Return ONLY a valid JSON array, no markdown, no explanation.
- Each element format: {
    "subject": "...",
    "subject_type": "ArchitecturalInnovation|TrainingObjective|LossFunction|HardwareConstraint|Dataset|Metric|Concept|Paper|Unknown",
    "subject_properties": {"summary": "...", "parameters": "..."},
    "predicate": "...",
    "object": "...",
    "object_type": "ArchitecturalInnovation|TrainingObjective|LossFunction|HardwareConstraint|Dataset|Metric|Concept|Paper|Unknown",
    "object_properties": {"summary": "...", "parameters": "..."},
    "confidence": 0.0-1.0
  }"""

BANNED_ENTITY_KEYWORDS = {
    "dataset", "novel dataset", "applicants", "million", "billion", 
    "percent", "results", "performance", "state of the art", "sota",
    "proposed method", "framework", "approach", "paper", "study", "author"
}

class LLMExtractor:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or self._default_client()

    def _default_client(self):
        return genai.GenerativeModel(getattr(cfg, "GEMINI_MODEL", "gemini-1.5-flash"))

    def _build_context_prompt(self, text: str, metadata: dict) -> str:
        pub_date = metadata.get("publication_date", "Unknown")
        citations = metadata.get("citation_count", 0)
        
        system_context = (
            f"--- PAPER METADATA ---\n"
            f"Publication Date: {pub_date}\n"
            f"Citation Count: {citations}\n"
            f"----------------------\n\n"
            f"Extract entities and high-signal semantic predicates from the text below. "
            f"Incorporate the chronological context when mapping generational dependencies between models or techniques."
        )
        return f"{_SYSTEM_PROMPT}\n\n{system_context}\n\nText:\n{text}"

    def extract(self, text: str, source_id: str = "", metadata: dict = None) -> list[Triple]:
        metadata = metadata or {}
        prompt = self._build_context_prompt(text, metadata)
        
        try:
            if hasattr(self.llm_client, "generate_content"):
                response = self.llm_client.generate_content(prompt)
                raw_text = response.text
            else:
                raw_text = self.llm_client.generate(prompt)

            # Clean potential markdown wrappers if the model ignores the instruction
            cleaned_text = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE)
            data = json.loads(cleaned_text)
            
            triples = []
            for item in data:
                triples.append(Triple(
                    subject=item.get("subject", ""),
                    predicate=item.get("predicate", "related_to"),
                    obj=item.get("object", ""),
                    confidence=float(item.get("confidence", 1.0)),
                    source_id=source_id,
                    subject_type=item.get("subject_type", "Unknown"),
                    obj_type=item.get("obj_type", "Unknown"),
                    subject_props=item.get("subject_properties", {}),
                    obj_props=item.get("object_properties", {})
                ))
            return triples
        except Exception as exc:
            logger.warning("LLM extraction failed for %s: %s", source_id, exc)
            return []

class MultiPassLLMExtractor(LLMExtractor):
    def extract(self, text: str, source_id: str = "", metadata: dict = None) -> list[Triple]:
        metadata = metadata or {}
        
        # Pass 1: Broad Candidate Extraction
        pass_one_prompt = (
            f"Pass 1: Extract all candidate entities (Models, Methods, Datasets, Metrics) "
            f"and relationships from the following text as a JSON array.\n\nText:\n{text}"
        )
        raw_extraction = self.llm_client.generate_content(pass_one_prompt).text if hasattr(self.llm_client, "generate_content") else self.llm_client.generate(pass_one_prompt)

        # Pass 2: Semantic Density Verification & Pruning
        pass_two_prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Pass 2: Review the following extracted candidates against the original text. "
            f"Discard superficial co-occurrences, isolate multi-hop dependencies, and ensure "
            f"only high-value structural contributions remain. Return ONLY the final JSON array.\n\n"
            f"Candidates:\n{raw_extraction}\n\nOriginal Text:\n{text}"
        )
        
        try:
            response_text = self.llm_client.generate_content(pass_two_prompt).text if hasattr(self.llm_client, "generate_content") else self.llm_client.generate(pass_two_prompt)
            cleaned_text = re.sub(r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.IGNORECASE)
            data = json.loads(cleaned_text)
            
            triples = []
            for item in data:
                triples.append(Triple(
                    subject=item.get("subject", ""),
                    predicate=item.get("predicate", "related_to"),
                    obj=item.get("object", ""),
                    confidence=float(item.get("confidence", 1.0)),
                    source_id=source_id,
                    subject_type=item.get("subject_type", "Unknown"),
                    obj_type=item.get("obj_type", "Unknown"),
                    subject_props=item.get("subject_properties", {}),
                    obj_props=item.get("object_properties", {})
                ))
            return triples
        except Exception as exc:
            logger.warning("Multi-pass extraction failed for %s: %s", source_id, exc)
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


def is_high_quality_entity(entity_name: str) -> bool:
    """
    Evaluates if an entity string meets the quality criteria to be stored in the graph.
    Blocks pure numbers, phrases below 3 chars, and generic qualitative filler phrases.
    """
    if not entity_name:
        return False
        
    clean_name = entity_name.strip()
    
    # 1. Minimum Length Check: Must be at least 3 characters
    if len(clean_name) < 3:
        logger.debug("Entity filtered (under 3 chars): '%s'", clean_name)
        return False
        
    # 2. Letter Verification: Must contain at least one literal letter
    if not re.search(r'[a-zA-Z]', clean_name):
        logger.debug("Entity filtered (no letters/pure numbers): '%s'", clean_name)
        return False
        
    # 3. Content-Quality Phrase Check: Lowercase scan against banned descriptive fragments
    lower_name = clean_name.lower()
    for phrase in BANNED_ENTITY_KEYWORDS:
        if phrase in lower_name:
            logger.debug("Entity filtered (banned descriptor matched '%s'): '%s'", phrase, clean_name)
            return False
            
    return True


def extract_triples(text: str, source_id: str = "") -> list[Triple]:
    """Convenience wrapper — uses whichever backend is configured and applies quality filters."""
    raw_triples = get_extractor().extract(text, source_id=source_id)
    
    high_quality_triples = []
    for t in raw_triples:
        # Ensure BOTH subject and object pass the quality filter
        if is_high_quality_entity(t.subject) and is_high_quality_entity(t.obj):
            high_quality_triples.append(t)
        else:
            logger.info("Dropping low-quality triple: (%s) -> [%s] -> (%s)", t.subject, t.predicate, t.obj)
            
    return high_quality_triples