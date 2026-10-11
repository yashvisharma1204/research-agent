"""
config.py — central config loaded from environment variables.
All modules import from here; nothing reads os.environ directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LLM_MODEL: str = "gemini-2.5-flash"                 # free-tier friendly

    # Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # Ingestion
    RESEARCH_TOPICS: list[str] = [
        t.strip()
        for t in os.getenv(
            "RESEARCH_TOPICS",
            "large language models,retrieval augmented generation,knowledge graphs",
        ).split(",")
    ]
    FETCH_INTERVAL_HOURS: int = int(os.getenv("FETCH_INTERVAL_HOURS", "24"))
    MAX_PAPERS_PER_TOPIC: int = int(os.getenv("MAX_PAPERS_PER_TOPIC", "20"))

    # Triple extraction backend: "llm" | "rebel" | "gliner"
    TRIPLE_EXTRACTION_METHOD: str = os.getenv("TRIPLE_EXTRACTION_METHOD", "llm")

    # Author handling. Authors are provenance metadata, not research content:
    # as embedded :Author entities they flood the graph + vector index and
    # crowd out concepts/models/results. When False (default) authors are stored
    # as a plain `authors` list property on the Paper node — no :Author nodes,
    # no name embeddings, no AUTHORED edges. Set EXTRACT_AUTHORS=true to restore
    # the full author/citation subgraph (still without name embeddings).
    EXTRACT_AUTHORS: bool = os.getenv("EXTRACT_AUTHORS", "false").lower() == "true"

    # Similarity threshold for entity resolution fuzzy merge
    ENTITY_MERGE_THRESHOLD: float = float(os.getenv("ENTITY_MERGE_THRESHOLD", "0.92"))

    # RAG retrieval
    GRAPH_HOP_LIMIT: int = int(os.getenv("GRAPH_HOP_LIMIT", "2"))
    VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "5"))
    GRAPH_TRIPLE_LIMIT: int = int(os.getenv("GRAPH_TRIPLE_LIMIT", "40"))

    # Seed entity resolution (query-time). When a query term like "LLM" doesn't
    # substring-match a node, resolve it against the entity_name_embedding vector
    # index so it still reaches "Large Language Model". This threshold is
    # intentionally LOOSER than ENTITY_MERGE_THRESHOLD (which governs destructive
    # ingest-time node merging): retrieval favours recall, so near-synonyms are
    # acceptable seeds, whereas merging two nodes must be near-certain.
    SEED_RESOLUTION_THRESHOLD: float = float(os.getenv("SEED_RESOLUTION_THRESHOLD", "0.55"))
    SEED_RESOLUTION_TOP_K: int = int(os.getenv("SEED_RESOLUTION_TOP_K", "3"))


cfg = Config()

if not cfg.GEMINI_API_KEY:
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "GEMINI_API_KEY is not set. LLM-backed features (routing, extraction, "
        "synthesis, summaries) will fail until it is configured in the environment "
        "or .env file."
    )

# High-signal semantic predicates & weighting for Layer 3 Graph Traversal
HIGH_SIGNAL_PREDICATES = {
    "replaces_component_with": 1.0,
    "introduces_training_objective": 0.95,
    "ameliorates_limitation_of": 0.90,
    "benchmarked_superior_to": 0.85,
    "proposes": 0.80,
    "uses_dataset": 0.60,
    "achieves_score_on": 0.60,
    "related_to": 0.30,
}


def get_predicate_weight(predicate: str) -> float:
    """Return the traversal weight for a given semantic predicate."""
    return HIGH_SIGNAL_PREDICATES.get(predicate, 0.40)