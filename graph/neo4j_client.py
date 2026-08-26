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
    # Canonical entity name uniqueness. The merger MERGEs on `name` only, so the
    # unique key must be `name` (not the (name, type) pair) to stay consistent.
    "CREATE CONSTRAINT entity_name IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.name IS UNIQUE",

    # Index on entity type for fast type-filtered queries and stats.
    "CREATE INDEX entity_type_idx IF NOT EXISTS "
    "FOR (e:Entity) ON (e.type)",

    # Full-text index for fuzzy entity search
    "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
    "FOR (e:Entity) ON EACH [e.name]",

    # Range indexes on relation properties for fast filtering / time queries.
    "CREATE INDEX relation_confidence IF NOT EXISTS "
    "FOR ()-[r:RELATION]-() ON (r.confidence)",

    "CREATE INDEX relation_last_seen IF NOT EXISTS "
    "FOR ()-[r:RELATION]-() ON (r.last_seen)",

    # Neo4j Native Vector Index
    "CREATE VECTOR INDEX entity_embedding IF NOT EXISTS "
    "FOR (e:Entity) ON (e.embedding) OPTIONS {indexConfig: {"
    "  `vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
]


def bootstrap_schema(driver: Driver) -> None:
    """Idempotently apply constraints and indexes."""
    with driver.session() as session:
        for stmt in SCHEMA_STATEMENTS:
            try:
                session.run(stmt)
            except Exception as exc:
                if "VECTOR INDEX" in stmt:
                    logger.warning("Optional vector index statement skipped: %s", exc)
                else:
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