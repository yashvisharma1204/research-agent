"""
graph/neo4j_client.py
Provides a singleton Neo4j driver and bootstraps the schema
(constraints + indexes) on first connect.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from neo4j import GraphDatabase, Driver

from config import cfg

logger = logging.getLogger(__name__)

# ── Schema Cypher ─────────────────────────────────────────────────────────────

SCHEMA_STATEMENTS = [
    # Uniqueness constraints (also create indexes)
    "CREATE CONSTRAINT entity_name_type IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",

    "CREATE CONSTRAINT paper_id IF NOT EXISTS "
    "FOR (p:Paper) REQUIRE p.url IS UNIQUE",

    # Full-text index for fuzzy entity search
    "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
    "FOR (e:Entity) ON EACH [e.name]",

    # Vector index placeholder (requires Neo4j 5.11+ with GDS)
    # Uncomment if using Neo4j native vector search:
    # "CREATE VECTOR INDEX entity_embedding IF NOT EXISTS "
    # "FOR (e:Entity) ON (e.embedding) OPTIONS {indexConfig: {"
    # "  `vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
]


def bootstrap_schema(driver: Driver) -> None:
    """Idempotently apply constraints and indexes."""
    with driver.session() as session:
        for stmt in SCHEMA_STATEMENTS:
            try:
                session.run(stmt)
            except Exception as exc:
                # Constraint may already exist — log and continue
                logger.debug("Schema statement skipped (%s): %s", exc, stmt[:60])
    logger.info("Neo4j schema bootstrapped")


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    """Return a cached Neo4j driver, bootstrapping schema on first call."""
    driver = GraphDatabase.driver(
        cfg.NEO4J_URI,
        auth=(cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD),
    )
    driver.verify_connectivity()
    logger.info("Connected to Neo4j at %s", cfg.NEO4J_URI)
    bootstrap_schema(driver)
    return driver