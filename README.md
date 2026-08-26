<div align="center">

# Self-Evolving Research Agent

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Database-Neo4j_Aura-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Vector Index](https://img.shields.io/badge/Vector_Index-Neo4j_Native-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-228B22?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![ONNX](https://img.shields.io/badge/Embeddings-ONNX_Runtime-005CBB?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)

<img width="812" height="541" alt="Self Evolving Research Agent" src="https://github.com/user-attachments/assets/ce20ad31-7171-4716-ba50-3a07ab22df79" />

</div>



---

## Executive Summary

The **Self-Evolving Research Agent (Research Workbench)** is a production-grade, continuous-learning hybrid Retrieval-Augmented Generation (RAG) platform. Traditional vector-only RAG pipelines suffer from context dilution, a lack of deterministic lineage, and an inability to perform multi-hop reasoning over connected domain concepts.

This platform bridges dense semantic vector retrieval with deterministic property graph topologies. It operates as an autonomous backend that ingests unstructured scientific literature (via arXiv, Semantic Scholar, RSS feeds, or raw payloads), parses semantic relationships into subject-predicate-object knowledge triples, builds an interconnected **Neo4j Knowledge Graph**, indexes dense embeddings directly in **Neo4j's native vector index** (computed locally via **ONNX Runtime**), and coordinates synthesis through **Google Gemini 2.5 Flash**.

---

## Theoretical Foundations & System Philosophy


```
                           ┌─────────────────────────┐
                           │  Scientific Literature  │
                           │  (arXiv, RSS, Payloads) │
                           └────────────┬────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │   FastAPI Orchestrator  │
                           └──────┬───────────┬──────┘
                                  │           │
       ┌──────────────────────────┘           └──────────────────────────┐
       ▼                                                                 ▼

┌───────────────────────────────────────┐                         ┌──────────────────────┐
│       Knowledge Graph Engine          │                         │ Dense Vector Engine  │
│  (Neo4j Deterministic Triples)        │                         │ (Neo4j Native Vector)│
└──────────────────┬────────────────────┘                         └──────────┬───────────┘
                   │                                                         │
                   │          ┌───────────────────────────────────┐          │
                   └─────────►│ Hybrid Traversal & Context Fusion │◄─────────┘
                              └─────────────────┬─────────────────┘
                                                │
                                                ▼
                              ┌───────────────────────────────────┐
                              │    Generative Synthesis Engine    │
                              │      (Google Gemini 2.5 Flash)    │
                              └───────────────────────────────────┘

```

### 1. Vector Search vs. Graph Traversal
Standard vector retrieval translates text blocks into embeddings $\mathbf{v} \in \mathbb{R}^d$ and performs nearest-neighbor search based on cosine similarity:

$$\text{Sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

While effective for unstructured semantic matching, vector similarity fails to represent **explicit, multi-step relational structures** (e.g., *Model A* `USES_METHOD` *Optimization B*, which `IMPROVES` *Metric C*).

The Research Workbench implements a dual retrieval model:
* **Probabilistic Semantic Search (Neo4j Native Vector Index):** Localizes relevant document chunks across high-dimensional semantic spaces using embeddings stored directly on graph nodes.
* **Deterministic Graph Traversal (Neo4j):** Executes variable-length Cypher path traversals to collect $n$-hop relation subgraphs surrounding candidate nodes.

### 2. Solving Catastrophic Forgetting & Static Indices
Static vector indices cannot natively adapt when new information contradicts or extends prior documents without complete re-indexing. By decoupling entity persistence from raw text, our architecture dynamically updates node attributes (such as `mention_count` and node degree) and merges relation edges idempotently.

### 3. Open Information Extraction (OpenIE) Formulation
Unstructured document text is converted into set representations of relational triples:

$$\mathcal{T} = \{ (s, p, o) \mid s \in \mathcal{E}, o \in \mathcal{E}, p \in \mathcal{R} \}$$

Where $\mathcal{E}$ represents the set of extracted entities and $\mathcal{R}$ represents the set of directed predicates. Each triple is assigned an extraction confidence score $c \in [0.0, 1.0]$.

---

## System Architecture & Micro-Layers

The system is decoupled into four functional execution layers:


```

┌──────────────────────────────────────────────────────────────────────────────────┐
│                                 FASTAPI ORCHESTRATOR                             │
│  - Async I/O Event Loop        - Connection Pooling       - Cache-Aside Manager  │
└──────────┬───────────────────────────────┬───────────────────────────────┬───────┘
           │                               │                               │
           ▼                               ▼                               ▼
┌────────────────────────┐        ┌───────────────────────┐       ┌──────────────────────┐
│ INGESTION & EXTRACTION │        │ HYBRID STORAGE ENGINE │       │ GENERATIVE AI LAYER  │
│ - arXiv / RSS Fetch    │        │ - Neo4j (Graph DB)    │       │ - Gemini 2.5 Flash   │
│ - ONNX Transformer     │        │ - Neo4j Vector Index  │       │ - Lazy Summary       │
│ - Triple Extraction    │        │ - Idempotent KG Merger│       │ - Sub-graph Synthesis│
└────────────────────────┘        └───────────────────────┘       └──────────────────────┘

```

### 1. The Orchestrator (FastAPI)
Acts as the asynchronous API gateway and state machine controller.
* **Concurrency Model:** Built on Python’s native `asyncio` event loop. Unblocking tasks (e.g., remote arXiv fetching, parallel LLM inference calls) are scheduled asynchronously.
* **Cache-Aside / Read-Through Pattern:** Incoming requests for paper summaries query the primary Neo4j node attributes. On a cache miss, execution routes to the LLM generation layer, streams the response back to the API route, and asynchronously executes a node mutation back-write.

### 2. Ingestion & Extraction Engine
Executes natural language processing to extract structural graph data from raw documents.
* **Fetchers:** Interacts with external REST/XML endpoints (arXiv API, PubMed, RSS feeds) via client session pooling.
* **Entity Identification & Disambiguation:** Parses extracted entities into unified canonical forms, preventing duplicate nodes for terms like *"LLMs"*, *"Large Language Models"*, and *"large language model"*.
* **Confidence Scoring:** Triples undergo threshold filtering. Extracted relationships with confidence scores below a configurable limit ($c < \tau$) are dropped prior to database insertion.

### 3. Hybrid Storage Engine
* **Neo4j Property Graph:** Maintains transactional integrity for the knowledge graph. Stores entities as nodes and predicates as directed edges. Handles real-time graph projections for visualization and multi-hop Cypher querying.
* **Neo4j Native Vector Index:** Dense embeddings are stored as an `embedding` property directly on each `:Entity` node and searched via Neo4j's built-in HNSW vector index (`db.index.vector.queryNodes`). Text payloads are vectorized locally using ONNX Runtime transformer models, avoiding API network latency during embedding creation. Keeping vectors in Neo4j means there is no separate index file to persist — a key advantage on ephemeral hosting (e.g. free-tier PaaS) where local files are wiped on restart.

### 4. Generative AI & Synthesis Layer (Google Gemini 2.5 Flash)
* **Structured Retrieval Context:** Merges vector search results with structural graph neighborhood sub-graphs into a unified prompt context.
* **Lazy Evaluation Summarizer:** Generates standardized 3-bullet point scientific summaries bound by strict system instruction schemas.
* **Self-Correction Audit:** A second Gemini pass audits the generated answer against the retrieved context for hallucinations and returns a faithfulness verdict.

---

## Knowledge Graph Schema & Ontology

The Neo4j database uses a custom scientific ontology designed to capture academic literature structure.


```
                       (:Entity:Paper)
                            │
           ┌────────────────┼────────────────┐
           │                │                │
  (INTRODUCES)        (USES_METHOD)     (BENCHMARKS)
           │                │                │
           ▼                ▼                ▼
    (:Entity:Model)  (:Entity:Method)  (:Entity:Metric)

```

### Node Constraints & Properties (`:Entity`)

Nodes represent core concepts, artifacts, authors, and methodologies.

| Property | Type | Indexing | Description |
| :--- | :--- | :--- | :--- |
| `name` | `STRING` | **Unique Constraint** | Canonical entity identifier (e.g., `"DistilBERT"`). |
| `type` | `STRING` | Key Index | Ontological type: `Paper`, `Method`, `Model`, `Metric`, `Author`, `Concept`, `Organisation`. |
| `mention_count` | `INTEGER` | Range Index | Frequency counter incremented on distinct paper ingestion. |
| `text` | `STRING` | None | Raw unstructured text snippet or original abstract payload. |
| `summary` | `STRING` | None | Cached 3-bullet markdown summary generated by Gemini. |
| `created_at` | `ZONED DATETIME` | None | UTC timestamp of initial graph insertion. |

### Edge Properties (`:RELATION`)

Directed relationships connect entities via extracted predicates.

| Property | Type | Description |
| :--- | :--- | :--- |
| `type` | `STRING` | Normalized verb phrase (e.g., `OPTIMIZES`, `INTRODUCES`, `EVALUATES_ON`). |
| `confidence` | `FLOAT` | Extraction confidence score derived from the NLP model ($0.0 \le c \le 1.0$). |
| `sources` | `LIST<STRING>` | Array of originating document IDs / DOIs ensuring provenance. |

### Cypher Schema Definitions
```cypher
// Enforce canonical entity name uniqueness
CREATE CONSTRAINT unique_entity_name IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE;

// Indexes for performance optimization
CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_mention_idx IF NOT EXISTS FOR (e:Entity) ON (e.mention_count);

```

---

## Data Pipeline Lifecycle

```
[User / Cron Trigger] ──► POST /ingest/arxiv
                                 │
                                 ▼
                    [Fetch Raw Abstract / Metadata]
                                 │
                                 ▼
                     [ONNX Text Vectorization]
                                 │
                                 ▼
                    [Extract Knowledge Triples]
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
[Neo4j Cypher Idempotent MERGE]             [Neo4j Vector Property Write]
  - MERGE (s:Entity {name: ...})              - Compute ONNX embedding
  - ON MATCH SET count = count + 1            - SET e.embedding = vector
  - MERGE (s)-[r:RELATION]->(o)               - Indexed by native vector index
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 │
                                 ▼
                    [Ready for Context Retrieval]
                                 │
                                 ▼
                   GET /api/paper/summary/{title}
                                 │
                 ┌───────────────┴───────────────┐
            (Cache Hit)                     (Cache Miss)
                 │                               │
                 ▼                               ▼
       [Return Node Summary]            [Query Gemini 2.5 LLM]
                                                 │
                                                 ▼
                                     [Write-Back to Neo4j Node]

```

### 1. Ingestion Phase

An ingestion job is initialized via API payload or worker trigger:

1. **Fetch & Normalize:** Raw text, metadata, and DOIs are retrieved.
2. **Text Chunking & Embedding:** The payload is processed using standard sliding-window tokenization and converted into dense vectors via ONNX runtime bindings.
3. **Extraction:** The text block is analyzed to generate candidate $(s, p, o)$ triples along with confidence metrics $c$.

### 2. Graph Merge Strategy (Idempotent Cypher Mutations)

To prevent network graph fragmentation and duplicate entity generation, updates are wrapped in transactional Cypher `MERGE` blocks:

```cypher
UNWIND $triples AS triple
MERGE (s:Entity {name: triple.subject})
  ON CREATE SET s.type = triple.subject_type, s.mention_count = 1
  ON MATCH SET s.mention_count = s.mention_count + 1

MERGE (o:Entity {name: triple.object})
  ON CREATE SET o.type = triple.object_type, o.mention_count = 1
  ON MATCH SET o.mention_count = o.mention_count + 1

MERGE (s)-[r:RELATION {type: triple.predicate}]->(o)
  ON CREATE SET r.confidence = triple.confidence, r.sources = [triple.source_id]
  ON MATCH SET r.sources = apoc.coll.toSet(r.sources + triple.source_id);

```

---

## API Reference Specification

All endpoints are served via FastAPI with automatic OpenAPI documentation available at `/docs`.

| Method | Endpoint | Query / Body Params | Response Type | Description |
| --- | --- | --- | --- | --- |
| **GET** | `/health` | None | `JSON` | Liveness probe. Returns `{ "status": "ok" }`. |
| **GET** | `/stats` | None | `JSON` | Aggregate graph metrics: entity count, relation count, paper count, vector count. |
| **POST** | `/query` | `{ "question": "string", "include_context": false }` | `JSON` | Hybrid graph-vector retrieval, then synthesized answer from Gemini with a faithfulness audit. |
| **POST** | `/ingest/arxiv` | `{ "arxiv_query": "string", "max_results": 5 }` | `JSON` | Fetches, parses, vectorizes, and merges arXiv literature into the graph. Requires ingestion password (HTTP Basic). |
| **POST** | `/ingest/text` | `{ "title": "string", "text": "string", "doc_id": "", "published_date": "Unknown" }` | `JSON` | Manual ingestion of raw text / notes. Requires ingestion password (HTTP Basic). |
| **GET** | `/graph` | `?limit=120` | `JSON` | Serialized `{ nodes, links }` dataset for D3.js force-directed rendering. |
| **GET** | `/api/paper/summary/{paper_title}` | Path Parameter: `paper_title` | `JSON` | Read-through cache lookup for a paper summary; lazily invokes Gemini on cache miss and writes it back to the node. |
| **GET** | `/api/ingestion/history` | None | `JSON` | Recent ingested documents (title, type, text length, published date), newest first, limited to 20. |

### API Payload Schema Examples

#### Request: `POST /query`

```json
{
  "question": "What optimization techniques are applied to DistilBERT models?",
  "include_context": false
}

```

#### Response: `POST /query`

```json
{
  "question": "What optimization techniques are applied to DistilBERT models?",
  "answer": "DistilBERT models primarily utilize Knowledge Distillation during pre-training, combined with 8-bit Quantization to reduce inference latency without significant accuracy degradation...",
  "entity_seeds": ["DistilBERT", "Knowledge Distillation"],
  "graph_triple_count": 12,
  "vector_result_count": 5,
  "route": "hybrid",
  "model": "gemini-2.5-flash",
  "is_faithful": true,
  "critique_notes": "None",
  "context": null
}

```

> When `include_context` is `true`, the `context` field is populated with the
> raw `graph_triples` and `vector_results` used to synthesize the answer.

---

## Frontend System Engineering & Graph Physics

The frontend UI (`index.html`) is designed for high-throughput visualization without the runtime overhead of complex JavaScript frameworks.
<img width="1512" height="827" alt="image" src="https://github.com/user-attachments/assets/1c261f2e-bde3-4190-bc87-6bbfbfb6f8b4" />



### 1. Bento UI Architecture

Structured using CSS Grid (`.workbench-grid`) to isolate visualization components:

* **Hushed Color Palette:** High-contrast neutral tones (`#F8F9FA`, `#1E293B`) reduce visual fatigue during graph analysis.
* **Responsive Layout:** Adjusts component density based on view size while maintaining visual hierarchy.

### 2. D3.js Force Simulation Dynamics

The visualization canvas uses `d3-force` to map graph database nodes to a 2D physics engine:

* **Node Repulsion (`forceManyBody`):** Applied with a negative strength scalar to prevent node overlap:
$$F_{\text{repulsion}} = \frac{-\gamma}{d^2}$$


* **Radial Categorical Gravity (`forceRadial`):** Pulls nodes toward distinct concentric radius bands based on their ontological `type`:
$$F_{\text{radial}} = \alpha \cdot (r_{\text{target}} - r_{\text{current}})$$


* **Dynamic Node Radii:** Node size scales relative to its structural importance using logarithmically bucketed degree frequencies:
$$R(n) = R_{\text{base}} + k \cdot \log(1 + \text{mentionCount}(n))$$


* **Ontological Palette Mapping:**
* `Paper` $\rightarrow$ Deep Indigo (`#4F46E5`)
* `Method` $\rightarrow$ Mint Green (`#10B981`)
* `Model` $\rightarrow$ Crimson Red (`#EF4444`)
* `Metric` $\rightarrow$ Amber Yellow (`#F59E0B`)



### 3. Inspector Drawer Mechanics

Clicking a node halts the local simulation execution loop to release CPU resources. It triggers a slide-out drawer (`.inspector-drawer`) that retrieves cached Gemini summaries from `/api/paper/summary/{title}` and presents extracted relation metrics.

---

## Environment Configuration & Setup

### Environment Variables (`.env`)

```bash
# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=YourSecurePassword

# Google Gemini API (required for routing, extraction, and synthesis)
GEMINI_API_KEY=your-gemini-api-key

# Ingestion auth — password-only. If unset, the /ingest/* endpoints are disabled.
INGESTION_PASSWORD=your-ingestion-password

# CORS — comma-separated allowed origins. Defaults to "*".
# Credentials are only enabled when explicit (non-wildcard) origins are set.
CORS_ALLOW_ORIGINS=*

# Ingestion tuning (optional)
RESEARCH_TOPICS=large language models,retrieval augmented generation,knowledge graphs
FETCH_INTERVAL_HOURS=24
MAX_PAPERS_PER_TOPIC=20
TRIPLE_EXTRACTION_METHOD=llm
ENTITY_MERGE_THRESHOLD=0.92

# RAG retrieval tuning (optional)
GRAPH_HOP_LIMIT=2
VECTOR_TOP_K=5
GRAPH_TRIPLE_LIMIT=40

```

> Embeddings are computed locally with ONNX Runtime (`all-MiniLM-L6-v2`, 384-dim)
> and stored directly on Neo4j nodes — there is no separate vector index file to
> configure or persist.

### Local Installation & Execution

```bash
# 1. Clone Repository
git clone [https://github.com/your-org/self-evolving-research-agent.git](https://github.com/your-org/self-evolving-research-agent.git)
cd self-evolving-research-agent

# 2. Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Start Neo4j Database Instance via Docker
docker run -d \
  --name research-agent-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/YourSecurePassword \
  neo4j:5.12-community

# 5. Launch FastAPI Backend Service
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

```

---
<div align="center">
Self-Evolving Research Agent • Built for Autonomous Scientific Knowledge Discovery
</div>
