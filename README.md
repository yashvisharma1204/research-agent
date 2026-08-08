<div align="center">

# Self-Evolving Research Agent

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Database-Neo4j_Aura-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![FAISS](https://img.shields.io/badge/Vector_Index-FAISS-7A8494?style=flat-square&logo=cpu&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.0_Flash-228B22?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![ONNX](https://img.shields.io/badge/Inference-ONNX_Runtime-005CBB?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)

<img width="812" height="541" alt="Self Evolving Research Agent" src="https://github.com/user-attachments/assets/ce20ad31-7171-4716-ba50-3a07ab22df79" />

</div>



---

## Executive Summary

The **Self-Evolving Research Agent (Research Workbench)** is a production-grade, continuous-learning hybrid Retrieval-Augmented Generation (RAG) platform. Traditional vector-only RAG pipelines suffer from context dilution, a lack of deterministic lineage, and an inability to perform multi-hop reasoning over connected domain concepts.

This platform bridges dense semantic vector retrieval with deterministic property graph topologies. It operates as an autonomous backend that ingests unstructured scientific literature (via arXiv, RSS feeds, or raw payloads), parses semantic relationships into subject-predicate-object knowledge triples, builds an interconnected **Neo4j Knowledge Graph**, indexes dense embeddings via **FAISS**, and coordinates synthesis through **Google Gemini 2.0 Flash**.

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
│  (Neo4j Deterministic Triples)        │                         │  (FAISS ONNX Embeds) │
└──────────────────┬────────────────────┘                         └──────────┬───────────┘
                   │                                                         │
                   │          ┌───────────────────────────────────┐          │
                   └─────────►│ Hybrid Traversal & Context Fusion │◄─────────┘
                              └─────────────────┬─────────────────┘
                                                │
                                                ▼
                              ┌───────────────────────────────────┐
                              │    Generative Synthesis Engine    │
                              │      (Google Gemini 2.0 Flash)    │
                              └───────────────────────────────────┘

```

### 1. Vector Search vs. Graph Traversal
Standard vector retrieval translates text blocks into embeddings $\mathbf{v} \in \mathbb{R}^d$ and performs nearest-neighbor search based on cosine similarity:

$$\text{Sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

While effective for unstructured semantic matching, vector similarity fails to represent **explicit, multi-step relational structures** (e.g., *Model A* `USES_METHOD` *Optimization B*, which `IMPROVES` *Metric C*).

The Research Workbench implements a dual retrieval model:
* **Probabilistic Semantic Search (FAISS):** Localizes relevant document chunks across high-dimensional semantic spaces.
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
│  - Async I/O Event Loop        - Connection Pooling       - Cache-Aside Manager │
└──────────┬───────────────────────────────┬───────────────────────────────┬───────┘
           │                               │                               │
           ▼                               ▼                               ▼
┌────────────────────────┐        ┌───────────────────────┐       ┌──────────────────────┐
│ INGESTION & EXTRACTION │        │ HYBRID STORAGE ENGINE │       │ GENERATIVE AI LAYER  │
│ - arXiv / RSS Fetch    │        │ - Neo4j (Graph DB)    │       │ - Gemini 2.0 Flash.  │
│ - ONNX Transformer     │        │ - FAISS (Vector DB)   │       │ - Lazy Summary       │
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
* **FAISS Vector Store:** Operates an Inverted File with Product Quantization (`IVF-PQ`) index. Text payloads are vectorized using local ONNX Runtime transformer models, avoiding API network latency during token embedding creation.

### 4. Generative AI & Synthesis Layer (Google Gemini 2.0 Flash)
* **Structured Retrieval Context:** Merges vector search results with structural graph neighborhood sub-graphs into a unified prompt context.
* **Lazy Evaluation Summarizer:** Generates standardized 3-bullet point scientific summaries bound by strict system instruction schemas.

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
[Neo4j Cypher Idempotent MERGE]             [FAISS Index Vector Add]
  - MERGE (s:Entity {name: ...})              - Compute embedding
  - ON MATCH SET count = count + 1            - Append to index
  - MERGE (s)-[r:RELATION]->(o)               - Save index state
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
       [Return Node Summary]            [Query Gemini 2.0 LLM]
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
| **GET** | `/health` | None | `JSON` | Returns database pool status, vector index memory consumption, and liveness status. |
| **GET** | `/stats` | None | `JSON` | Aggregates database state metrics (total nodes, relation count, density metrics). |
| **POST** | `/query` | `{ "query": "string", "top_k": 5 }` | `JSON` | Performs hybrid graph-vector contextual retrieval and streams synthesized answer from Gemini. |
| **POST** | `/ingest/arxiv` | `{ "search_query": "string", "max_results": 10 }` | `JSON` | Triggers background worker to fetch, parse, vectorize, and merge arXiv literature into graph. |
| **POST** | `/ingest/text` | `{ "title": "string", "text": "string" }` | `JSON` | Manual ingestion payload endpoint for unindexed text, notes, or internal scientific reports. |
| **GET** | `/graph` | `?limit=100&min_mentions=1` | `JSON` | Streams serialized node/edge datasets formatted for D3.js force-directed rendering. |
| **GET** | `/api/paper/summary/{title}` | Path Parameter: `title` | `JSON` | Executes read-through cache lookup for paper summary; lazily invokes Gemini on cache miss. |
| **GET** | `/api/ingestion/history` | `?page=1&limit=20` | `JSON` | Returns audit log of ingested documents, success metrics, and parsed triple counts. |

### API Payload Schema Examples

#### Request: `POST /query`

```json
{
  "query": "What optimization techniques are applied to DistilBERT models?",
  "top_k": 5,
  "enable_graph_traversal": true
}

```

#### Response: `POST /query`

```json
{
  "query": "What optimization techniques are applied to DistilBERT models?",
  "answer": "DistilBERT models primarily utilize Knowledge Distillation during pre-training, combined with 8-bit Quantization (OPT-Q) to reduce inference latency without significant accuracy degradation...",
  "retrieved_nodes": [
    { "name": "DistilBERT", "type": "Model", "mention_count": 14 },
    { "name": "Knowledge Distillation", "type": "Method", "mention_count": 42 }
  ],
  "graph_paths": [
    "(:Model {name: 'DistilBERT'})-[:USES_METHOD]->(:Method {name: 'Knowledge Distillation'})"
  ],
  "latency_ms": 342.1
}

```

---

## Frontend System Engineering & Graph Physics

The frontend UI (`index.html`) is designed for high-throughput visualization without the runtime overhead of complex JavaScript frameworks.

<img width="1512" height="864" alt="Screenshot 2026-08-08 at 3 48 47 PM" src="https://github.com/user-attachments/assets/010ca1e5-2fde-4621-b89f-30632fb20675" />

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
# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=YourSecurePassword

# Google Gemini API
GEMINI_API_KEY=AIzaSy...

# Vector Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=./data/faiss_index.bin

```

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
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

```

---
<div align="center">
Self-Evolving Research Agent • Built for Autonomous Scientific Knowledge Discovery
</div>
