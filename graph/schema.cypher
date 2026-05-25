// schema.cypher — run once to set up Neo4j constraints and indexes

// Uniqueness constraints (also implicitly create indexes)
CREATE CONSTRAINT entity_name_type IF NOT EXISTS
FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE;

CREATE CONSTRAINT paper_id IF NOT EXISTS
FOR (p:Paper) REQUIRE p.url IS UNIQUE;

// Full-text index for fuzzy entity search
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (n:Entity) ON EACH [n.name];

// Range index on confidence for fast triple filtering
CREATE INDEX relation_confidence IF NOT EXISTS
FOR ()-[r:RELATION]-() ON (r.confidence);

// Range index for time-based queries
CREATE INDEX relation_last_seen IF NOT EXISTS
FOR ()-[r:RELATION]-() ON (r.last_seen);

// Optional: Neo4j 5.11+ native vector index (uncomment if using GDS)
// CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
// FOR (e:Entity) ON (e.embedding)
// OPTIONS {indexConfig: {
//   `vector.dimensions`: 384,
//   `vector.similarity_function`: 'cosine'
// }};
