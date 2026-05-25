"""
config.py — central config loaded from environment variables.
All modules import from here; nothing reads os.environ directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Gemini
    GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
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

    # Similarity threshold for entity resolution fuzzy merge
    ENTITY_MERGE_THRESHOLD: float = float(os.getenv("ENTITY_MERGE_THRESHOLD", "0.92"))

    # RAG retrieval
    GRAPH_HOP_LIMIT: int = int(os.getenv("GRAPH_HOP_LIMIT", "2"))
    VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "5"))
    GRAPH_TRIPLE_LIMIT: int = int(os.getenv("GRAPH_TRIPLE_LIMIT", "40"))


cfg = Config()
