// schema.cypher — run once to set up Neo4j constraints and indexes.
// This mirrors SCHEMA_STATEMENTS in graph/neo4j_client.py.

// Canonical entity name uniqueness. Entities are MERGEd on `name` only,
// so the unique key is `name` (not the (name, type) pair).
CREATE CONSTRAINT entity_name IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE;

// Index on entity type for fast type-filtered queries and stats.
CREATE INDEX entity_type_idx IF NOT EXISTS
FOR (e:Entity) ON (e.type);

// Full-text index for fuzzy entity search
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name];

// Range index on confidence for fast triple filtering
CREATE INDEX relation_confidence IF NOT EXISTS
FOR ()-[r:RELATION]-() ON (r.confidence);

// Range index for time-based queries
CREATE INDEX relation_last_seen IF NOT EXISTS
FOR ()-[r:RELATION]-() ON (r.last_seen);

// Neo4j 5.11+ native vector index for ENTITY RESOLUTION (merge/clustering).
// Holds a short vector of the entity *name* only, on `:Entity.name_embedding`.
// Kept separate from retrieval vectors so name-similarity and passage-similarity
// never pollute each other. (384-dim, cosine — matches all-MiniLM-L6-v2.)
CREATE VECTOR INDEX entity_name_embedding IF NOT EXISTS
FOR (e:Entity) ON (e.name_embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};

// ── Chunk-level semantic retrieval ──────────────────────────────────────────
// Passages of ingested documents are stored as :Chunk nodes linked to their
// source :Entity (Paper/Document) via [:FROM_DOC]. Retrieval queries the chunk
// vector index for passage-level precision, then resolves the parent document.
CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}};
