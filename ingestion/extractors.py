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

from config import cfg
from utils.gemini import get_client

logger = logging.getLogger(__name__)


# A URL/DOI is provenance (belongs on :Paper.url or a :CITES edge), never a
# first-class semantic entity. The LLM occasionally emits raw links (e.g.
# "http://arxiv.org/abs/1706.03762") as a triple subject/object; those collapse
# into meaningless, un-dedupable nodes in the graph. Drop them at the source.
_URL_RE = re.compile(r"^\s*(?:https?://|www\.|ftp://|doi:|10\.\d{4,}/)", re.IGNORECASE)


def _is_url_like(value: str) -> bool:
    """True if *value* looks like a bare URL/DOI rather than a named entity."""
    if not value:
        return False
    v = value.strip()
    if _URL_RE.match(v):
        return True
    # Catch bare domains with a path, e.g. "arxiv.org/abs/1706.03762", that lack
    # a scheme but are still just links, not entities.
    if re.match(r"^[\w.-]+\.(?:org|com|net|edu|gov|io|ai)/\S+$", v, re.IGNORECASE):
        return True
    return False


def _valid_entity_name(value: str) -> bool:
    """A usable entity name is non-empty and not a bare URL/DOI."""
    return bool(value and value.strip()) and not _is_url_like(value)


# Self-referential anaphors the LLM uses instead of naming the paper, e.g.
# "This paper proposes T5". If left as-is they're junk entities (rejected by the
# quality filter), which silently discards the paper's own highest-signal
# contribution triples. Rewriting them to the paper title both saves the triple
# AND anchors the paper node to what it proposes/introduces.
_SELF_REF_ANAPHORS = {
    "this paper", "the paper", "our paper", "this work", "the work",
    "our work", "this study", "the study", "our study", "this approach",
    "our approach", "the proposed method", "our method", "this method",
    "the authors", "we", "our model", "the model proposed", "present work",
    "this article", "the article", "our contribution",
}


def _canonicalise_self_reference(name: str, paper_title: str) -> str:
    """Rewrite a self-referential anaphor ("This paper", "We", ...) to the paper
    title so the triple survives and anchors the paper. Non-anaphors pass through
    unchanged. No-op when no paper_title is known."""
    if not paper_title:
        return name
    if (name or "").strip().lower().rstrip(".") in _SELF_REF_ANAPHORS:
        return paper_title
    return name


@dataclass
class Triple:
    subject: str
    predicate: str
    obj: str
    confidence: float = 1.0
    source_id: str = ""
    subject_type: str = "Unknown"
    obj_type: str = "Unknown"
    subject_props: dict = None
    obj_props: dict = None
    temporal_rationale: str = ""  # <--- Added temporal rationale property

    def __post_init__(self):
        self.subject_props = self.subject_props or {}
        self.obj_props = self.obj_props or {}

    def __repr__(self):
        return f"({self.subject}:{self.subject_type}) --[{self.predicate}]--> ({self.obj}:{self.obj_type})"


@dataclass
class ScoreResult:
    """A reified quantitative result: <model> achieves <value> <metric> on <dataset>.

    Captures evidence-bearing benchmark claims far better than a flat stringy
    triple. Written as (:Model)-[:ACHIEVES]->(:Result)-[:ON]->(:Dataset) and
    (:Result)-[:MEASURED_BY]->(:Metric).
    """
    model: str
    dataset: str
    metric: str
    value: str = ""
    source_id: str = ""

    def is_valid(self) -> bool:
        return bool(self.model.strip() and self.dataset.strip() and self.metric.strip())


@dataclass
class Contribution:
    """A reified *contribution* of a paper: the thing the paper actually did.

    Flat triples shatter a paper's real contribution into disconnected Concept
    nodes joined by weak `related_to` edges, which the retriever scores as
    noise. A Contribution captures the whole claim as one navigable structure:

        (:Paper)-[:MAKES_CLAIM]->(:Contribution {statement, kind})
        (:Contribution)-[:ADDRESSES]->(target)   # the problem/task tackled
        (:Contribution)-[:VIA]->(mechanism)       # how it was tackled

    `kind` is a controlled vocabulary so contributions are filterable:
    proposes_method | fixes_limitation | improves_metric | new_dataset.
    """
    statement: str
    kind: str = "proposes_method"
    addresses: str = ""          # Task / Concept the contribution tackles
    addresses_type: str = "Concept"
    via: str = ""                # mechanism: ArchitecturalInnovation / TrainingObjective
    via_type: str = "ArchitecturalInnovation"
    source_id: str = ""

    _VALID_KINDS = {"proposes_method", "fixes_limitation", "improves_metric", "new_dataset"}

    def is_valid(self) -> bool:
        # A contribution is only useful if it has a statement AND at least one
        # structural anchor (what it addresses or the mechanism it uses).
        return bool(self.statement.strip()) and bool(self.addresses.strip() or self.via.strip())

    def normalised_kind(self) -> str:
        return self.kind if self.kind in self._VALID_KINDS else "proposes_method"


# <--- Updated Prompt with Temporal Anchor Rules --->
_SYSTEM_PROMPT = """You are an advanced scientific knowledge graph extractor.
Given a text, extract factual (subject, predicate, object) triples and rich property payloads.

Rules:
- Subjects and objects should be granular technical entities or mechanism components.
- Classify every subject and object into ONE of these types:
    Model                   — named ML models / systems (e.g., BERT, GPT-4, Llama 2, ResNet)
    ArchitecturalInnovation — structural design changes (e.g., Rotary Embedding, SwiGLU, Multi-Query Attention)
    TrainingObjective       — loss formulations, pre-training tasks (e.g., Masked Language Modeling, DPO, RLHF)
    LossFunction            — specific mathematical losses (e.g., Cross-Entropy, Contrastive Loss)
    HardwareConstraint      — compute/memory constraints (e.g., Tensor Parallelism, FlashAttention-2)
    Dataset                 — benchmarks and corpora (e.g., SQuAD, MMLU, CommonCrawl)
    Metric                  — evaluation metrics (e.g., BLEU, Perplexity, Accuracy)
    Task                    — problem settings a model addresses (e.g., Question Answering, Summarization, Image Classification)
    Concept                 — foundational abstract ideas (e.g., Hallucination, Weight Quantization)
    Organisation            — labs, universities, companies (e.g., Google DeepMind, OpenAI, Stanford)
    Paper                   — publication references
    Unknown                 — fallback type
- Do NOT extract individual people / author names as entities. Author names are
  provenance metadata handled elsewhere, not research content. If a contribution
  must be attributed, attribute it to the Organisation, never a person.
- Prefer the most specific type. Use Model for a named system; use ArchitecturalInnovation for one of its internal components; use Task for the problem it solves.
- Extract a "properties" dictionary for each node containing technical attributes (e.g., parameters, dimensions, brief operational summaries if mentioned).
- Predicates should be high-signal semantic verbs: replaces_component_with, introduces_training_objective, ameliorates_limitation_of, benchmarked_superior_to, proposes, uses_dataset, achieves_score_on, evaluated_on_task, developed_by, related_to.

### GENERATIONAL SCHEMA ###
You MUST use the provided Publication Date to anchor relative temporal claims. Use these specific predicates for generational mappings:
- BUILDS_UPON: The current text extends an older architecture or dataset.
- SUPERSEDES: The current text replaces, outperforms, or deprecates a previous method.
- DERIVED_FROM: The model or data originates from a prior source.

Return ONLY a valid JSON object (no markdown, no explanation) with three keys, "triples", "results", and "contributions".

"triples" is an array where each element has the format: {
    "subject": "...",
    "subject_type": "Model|ArchitecturalInnovation|TrainingObjective|LossFunction|HardwareConstraint|Dataset|Metric|Task|Concept|Author|Organisation|Paper|Unknown",
    "subject_properties": {"summary": "...", "parameters": "..."},
    "predicate": "...",
    "object": "...",
    "object_type": "Model|ArchitecturalInnovation|TrainingObjective|LossFunction|HardwareConstraint|Dataset|Metric|Task|Concept|Author|Organisation|Paper|Unknown",
    "object_properties": {"summary": "...", "parameters": "..."},
    "confidence": 0.0-1.0,
    "temporal_rationale": "Brief explanation of the timeline dependency if BUILDS_UPON, SUPERSEDES, or DERIVED_FROM is used (otherwise leave empty)"
  }

"results" is an array of quantitative benchmark outcomes stated in the text (may be empty). Each element captures "<model> achieves <value> <metric> on <dataset>": {
    "model": "name of the model/system that achieved the score",
    "dataset": "benchmark or dataset the score was measured on",
    "metric": "evaluation metric (e.g., Accuracy, F1, BLEU)",
    "value": "the numeric score as stated (e.g., '88.5', '92.3%')"
  }
Only include a result when the text explicitly reports a model's score on a named dataset/benchmark. Do not invent numbers.

"contributions" is an array (may be empty) capturing the PRIMARY contributions the paper itself makes — the things it actually proposes or achieves, not background facts. This is the most important output: it is what makes the paper's content retrievable, so prefer 1-4 precise contributions over many vague ones. Each element: {
    "statement": "one concise sentence describing the contribution in the paper's own terms",
    "kind": "proposes_method | fixes_limitation | improves_metric | new_dataset",
    "addresses": "the Task or Concept (the PROBLEM) this contribution tackles, e.g. 'Long-Range Dependency Modeling'",
    "addresses_type": "Task|Concept",
    "via": "the specific mechanism used (an ArchitecturalInnovation / TrainingObjective / Model component), e.g. 'Self-Attention'",
    "via_type": "ArchitecturalInnovation|TrainingObjective|Model|LossFunction"
  }
Rules for contributions:
- Only extract contributions the TEXT itself makes; do not attribute other papers' work.
- 'addresses' is the problem; 'via' is the solution mechanism. At least one of the two must be present.
- Keep 'statement' factual and specific — no marketing language."""

BANNED_ENTITY_KEYWORDS = {
    "dataset", "novel dataset", "applicants", "million", "billion", 
    "percent", "results", "performance", "state of the art", "sota",
    "proposed method", "framework", "approach", "paper", "study", "author"
}


def _safe_parse_json_array(text: str) -> list:
    """Parse a JSON array, recovering complete objects if the output was truncated."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find the last complete object by locating the last '}' before the truncation
        last_close = text.rfind("}")
        if last_close == -1:
            raise
        truncated = text[:last_close + 1] + "]"
        # Strip any trailing comma before the closing bracket
        truncated = re.sub(r",\s*]", "]", truncated)
        return json.loads(truncated)


def _safe_parse_payload(text: str) -> tuple[list, list]:
    """Parse the extractor output into (triples, results).

    Backward compatible: accepts either the new object form
    {"triples": [...], "results": [...]} or a bare JSON array (legacy), in which
    case the array is treated as triples and results is empty.
    """
    stripped = text.lstrip()
    if stripped.startswith("["):
        # Legacy bare-array form.
        return _safe_parse_json_array(text), []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to recover a truncated object by closing it.
        last_close = text.rfind("}")
        if last_close == -1:
            raise
        candidate = text[: last_close + 1]
        # Balance braces crudely by appending closers until it parses.
        for _ in range(3):
            try:
                obj = json.loads(candidate)
                break
            except json.JSONDecodeError:
                candidate += "}"
        else:
            # Last resort: pull the triples array out with the array parser.
            return _safe_parse_json_array(text), []
    if isinstance(obj, list):
        return obj, []
    triples = obj.get("triples", []) if isinstance(obj, dict) else []
    results = obj.get("results", []) if isinstance(obj, dict) else []
    return triples, results


def _parse_results(results_data: list, source_id: str) -> list[ScoreResult]:
    """Convert raw result dicts into validated ScoreResult objects."""
    out: list[ScoreResult] = []
    for item in results_data or []:
        if not isinstance(item, dict):
            continue
        r = ScoreResult(
            model=str(item.get("model", "") or "").strip(),
            dataset=str(item.get("dataset", "") or "").strip(),
            metric=str(item.get("metric", "") or "").strip(),
            value=str(item.get("value", "") or "").strip(),
            source_id=source_id,
        )
        if r.is_valid():
            out.append(r)
    return out


def _parse_contributions(contrib_data: list, source_id: str) -> list[Contribution]:
    """Convert raw contribution dicts into validated Contribution objects."""
    out: list[Contribution] = []
    for item in contrib_data or []:
        if not isinstance(item, dict):
            continue
        c = Contribution(
            statement=str(item.get("statement", "") or "").strip(),
            kind=str(item.get("kind", "") or "proposes_method").strip(),
            addresses=str(item.get("addresses", "") or "").strip(),
            addresses_type=str(item.get("addresses_type", "") or "Concept").strip() or "Concept",
            via=str(item.get("via", "") or "").strip(),
            via_type=str(item.get("via_type", "") or "ArchitecturalInnovation").strip() or "ArchitecturalInnovation",
            source_id=source_id,
        )
        if c.is_valid():
            out.append(c)
    return out


def _contributions_from_text(cleaned_text: str) -> list:
    """Pull the raw "contributions" array out of an extractor payload.

    Kept separate from _safe_parse_payload so that function's (triples, results)
    signature and its callers/tests stay unchanged. Best-effort: any parse
    failure yields an empty list (contributions are additive, never required).
    """
    stripped = cleaned_text.lstrip()
    if stripped.startswith("["):
        return []  # legacy bare-array form has no contributions
    try:
        obj = json.loads(cleaned_text)
    except json.JSONDecodeError:
        last_close = cleaned_text.rfind("}")
        if last_close == -1:
            return []
        candidate = cleaned_text[: last_close + 1]
        for _ in range(3):
            try:
                obj = json.loads(candidate)
                break
            except json.JSONDecodeError:
                candidate += "}"
        else:
            return []
    return obj.get("contributions", []) if isinstance(obj, dict) else []


class LLMExtractor:
    def __init__(self, llm_client=None):
        # `llm_client` may be injected (e.g. in tests) exposing a `.generate(prompt)`
        # method. By default we use the shared google-genai client.
        self.llm_client = llm_client
        self._client = None if llm_client is not None else get_client()
        self.last_results: list[ScoreResult] = []
        self.last_contributions: list[Contribution] = []

    def _generate(self, prompt: str, max_output_tokens: int = 8192) -> str:
        """Run a single generation and return the raw text."""
        from google.genai import types

        if self.llm_client is not None:
            # Custom injected client (test/mocks).
            if hasattr(self.llm_client, "generate_content"):
                return self.llm_client.generate_content(prompt).text
            return self.llm_client.generate(prompt)

        response = self._client.models.generate_content(
            model=getattr(cfg, "LLM_MODEL", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=max_output_tokens),
        )
        return response.text or ""

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
        self.last_results: list[ScoreResult] = []
        self.last_contributions: list[Contribution] = []

        try:
            raw_text = self._generate(prompt, max_output_tokens=8192)

            cleaned_text = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE)
            data, results_data = _safe_parse_payload(cleaned_text)

            triples = []
            for item in data:
                subject = item.get("subject", "")
                obj = item.get("object", "")
                # Drop triples whose endpoints are bare URLs/DOIs or empty — they
                # are provenance, not semantic entities, and only create
                # meaningless un-dedupable nodes in the graph.
                if not (_valid_entity_name(subject) and _valid_entity_name(obj)):
                    logger.debug(
                        "Skipping URL/empty triple for %s: (%r)-[%s]->(%r)",
                        source_id, subject, item.get("predicate"), obj,
                    )
                    continue
                triples.append(Triple(
                    subject=subject,
                    predicate=item.get("predicate", "related_to"),
                    obj=obj,
                    confidence=float(item.get("confidence", 1.0)),
                    source_id=source_id,
                    subject_type=item.get("subject_type", "Unknown"),
                    obj_type=item.get("object_type", "Unknown"),
                    subject_props=item.get("subject_properties", {}),
                    obj_props=item.get("object_properties", {}),
                    temporal_rationale=item.get("temporal_rationale", "") # <--- Added extraction mapping
                ))
            self.last_results = _parse_results(results_data, source_id)
            self.last_contributions = _parse_contributions(
                _contributions_from_text(cleaned_text), source_id
            )
            return triples
        except Exception as exc:
            logger.warning("LLM extraction failed for %s: %s", source_id, exc)
            return []

class MultiPassLLMExtractor(LLMExtractor):
    def extract(self, text: str, source_id: str = "", metadata: dict = None) -> list[Triple]:
        metadata = metadata or {}
        self.last_results: list[ScoreResult] = []
        self.last_contributions: list[Contribution] = []
        
        # Pass 1: Broad Candidate Extraction
        pass_one_prompt = (
            f"Pass 1: Extract all candidate entities (Models, Methods, Datasets, Metrics) "
            f"and relationships from the following text as a JSON array.\n\nText:\n{text}"
        )
        raw_extraction = self._generate(pass_one_prompt)

        # Pass 2: Semantic Density Verification & Pruning
        pass_two_prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Pass 2: Review the following extracted candidates against the original text. "
            f"Discard superficial co-occurrences, isolate multi-hop dependencies, and ensure "
            f"only high-value structural contributions remain. Return ONLY the final JSON array.\n\n"
            f"Candidates:\n{raw_extraction}\n\nOriginal Text:\n{text}"
        )
        
        try:
            response_text = self._generate(pass_two_prompt, max_output_tokens=8192)
            cleaned_text = re.sub(r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.IGNORECASE)
            data, results_data = _safe_parse_payload(cleaned_text)

            triples = []
            for item in data:
                subject = item.get("subject", "")
                obj = item.get("object", "")
                # Drop triples whose endpoints are bare URLs/DOIs or empty — they
                # are provenance, not semantic entities, and only create
                # meaningless un-dedupable nodes in the graph.
                if not (_valid_entity_name(subject) and _valid_entity_name(obj)):
                    logger.debug(
                        "Skipping URL/empty triple for %s: (%r)-[%s]->(%r)",
                        source_id, subject, item.get("predicate"), obj,
                    )
                    continue
                triples.append(Triple(
                    subject=subject,
                    predicate=item.get("predicate", "related_to"),
                    obj=obj,
                    confidence=float(item.get("confidence", 1.0)),
                    source_id=source_id,
                    subject_type=item.get("subject_type", "Unknown"),
                    obj_type=item.get("object_type", "Unknown"),
                    subject_props=item.get("subject_properties", {}),
                    obj_props=item.get("object_properties", {}),
                    temporal_rationale=item.get("temporal_rationale", "") # <--- Added extraction mapping
                ))
            self.last_results = _parse_results(results_data, source_id)
            self.last_contributions = _parse_contributions(
                _contributions_from_text(cleaned_text), source_id
            )
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

    def extract(self, text: str, source_id: str = "", metadata: dict = None) -> list[Triple]:
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

    def extract(self, text: str, source_id: str = "", metadata: dict = None) -> list[Triple]:
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

    # 0. URL/DOI rejection: a bare link is provenance (belongs on :Paper.url or
    #    a :CITES edge), not a semantic entity. Covers every backend in one spot.
    if _is_url_like(clean_name):
        logger.debug("Entity filtered (URL/DOI): '%s'", clean_name)
        return False
    
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
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, lower_name):
            logger.debug("Entity filtered (banned descriptor matched '%s'): '%s'", phrase, clean_name)
            return False
            
    return True


def extract_triples(text: str, source_id: str = "", published_date: str = "Unknown",
                    paper_title: str = "") -> list[Triple]:
    """Convenience wrapper — uses whichever backend is configured and applies quality filters."""
    triples, _results = extract_triples_and_results(text, source_id, published_date, paper_title)
    return triples


def extract_triples_and_results(
    text: str, source_id: str = "", published_date: str = "Unknown",
    paper_title: str = "",
) -> tuple[list[Triple], list[ScoreResult]]:
    """Extract quality-filtered triples AND reified quantitative results in one pass.

    Backward-compatible 2-tuple wrapper around
    extract_triples_results_contributions (drops the contributions). Existing
    callers that don't yet consume contributions keep working unchanged.
    """
    triples, results, _contribs = extract_triples_results_contributions(
        text, source_id, published_date, paper_title
    )
    return triples, results


def extract_triples_results_contributions(
    text: str, source_id: str = "", published_date: str = "Unknown",
    paper_title: str = "",
) -> tuple[list[Triple], list[ScoreResult], list[Contribution]]:
    """Extract quality-filtered triples, reified results, AND reified
    contributions in a SINGLE extraction pass.

    Contributions capture what a paper actually does; they are only kept when
    their structural anchors (addresses / via) survive the entity quality
    filter, keeping the reified subgraph clean — same discipline as results.
    """
    metadata = {"publication_date": published_date}

    extractor = get_extractor()
    raw_triples = extractor.extract(text, source_id=source_id, metadata=metadata)

    high_quality_triples = []
    for t in raw_triples:
        # Rewrite self-referential anaphors ("This paper", "We", ...) to the
        # paper title so the paper's own contribution triples survive the quality
        # filter and anchor the paper node, instead of being silently dropped.
        t.subject = _canonicalise_self_reference(t.subject, paper_title)
        t.obj = _canonicalise_self_reference(t.obj, paper_title)

        # Defensive author drop: even if the LLM ignores the prompt and emits an
        # Author-typed node, keep it out of the graph unless EXTRACT_AUTHORS.
        if not cfg.EXTRACT_AUTHORS and "Author" in (t.subject_type, t.obj_type):
            logger.debug("Dropping author triple: %s", t)
            continue
        if is_high_quality_entity(t.subject) and is_high_quality_entity(t.obj):
            high_quality_triples.append(t)
        else:
            logger.info("Dropping low-quality triple: (%s) -> [%s] -> (%s)", t.subject, t.predicate, t.obj)

    raw_results = getattr(extractor, "last_results", []) or []
    quality_results = [
        r for r in raw_results
        if is_high_quality_entity(r.model) and is_high_quality_entity(r.dataset) and is_high_quality_entity(r.metric)
    ]

    raw_contribs = getattr(extractor, "last_contributions", []) or []
    quality_contribs = []
    for c in raw_contribs:
        # Rewrite self-referential anchors to the title (same reason as triples).
        c.addresses = _canonicalise_self_reference(c.addresses, paper_title)
        c.via = _canonicalise_self_reference(c.via, paper_title)
        # Statement itself isn't an entity, but the structural anchors are nodes
        # in the graph — require any present anchor to pass the quality filter.
        addresses_ok = (not c.addresses) or is_high_quality_entity(c.addresses)
        via_ok = (not c.via) or is_high_quality_entity(c.via)
        if c.is_valid() and addresses_ok and via_ok:
            quality_contribs.append(c)
        else:
            logger.info("Dropping low-quality contribution: %s", c.statement[:60])

    return high_quality_triples, quality_results, quality_contribs