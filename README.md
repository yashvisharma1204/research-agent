
<div align="center">

# Self-Evolving Research Agent

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Database-Neo4j_Aura-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![FAISS](https://img.shields.io/badge/Vector_Index-FAISS-7A8494?style=flat-square&logo=cpu&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.0_Flash-228B22?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)

<img width="812" height="541" alt="Self Evolving Research Agent" src="https://github.com/user-attachments/assets/ce20ad31-7171-4716-ba50-3a07ab22df79" />

*A continuous-learning hybrid RAG system and knowledge graph orchestrator.*

</div>

**[Home](#)** | [Usage](USAGE.md) | [Architecture](ARCHITECTURE.md) | [Contributing]()

---

## 📑 Table of Contents
1. [System Overview](#1-system-overview)
2. [Architectural Deep Dive](#2-architectural-deep-dive)
3. [Graph Database Schema](#3-graph-database-schema)
4. [Comprehensive API Reference](#4-comprehensive-api-reference)
5. [Frontend Engineering](#5-frontend-engineering)
6. [Local Setup & Deployment](#6-local-setup--deployment)

---

## 1. System Overview

Most Retrieval-Augmented Generation (RAG) pipelines operate on static, point-in-time document embeddings, limiting their ability to synthesize complex, multi-hop scientific relationships. 

This project implements a **continuous-learning backend**. It leverages a FastAPI orchestrator to fetch external literature (e.g., arXiv papers on topics like "Model Distillation BERT"), parses the unstructured text into semantic triples, and writes them to a live Neo4j Knowledge Graph. When queried, it utilizes a hybrid routing strategy—combining vector similarity with deterministic Cypher traversals—to feed highly contextualized sub-graphs into Google Gemini for final synthesis.

---

## 2. Architectural Deep Dive

The system is strictly decoupled into ingestion, storage, retrieval, and presentation layers.

```text
 ┌───────────────────────┐       ┌────────────────────────┐
 │   External Sources    │       │   Extraction Pipeline  │
 │  (arXiv, RSS, Text)   │ ────> │ (NLP Triple Parsing)   │
 └───────────────────────┘       └───────────┬────────────┘
                                             │
 ┌───────────────────────┐                   ▼
 │  Frontend Client UI   │       ┌────────────────────────┐
 │ (D3.js Force Graph,   │ <──── │   FastAPI Orchestrator │
 │  Bento Dashboard)     │ ────> │   (Query & Synthesis)  │
 └───────────────────────┘       └───────────┬────────────┘
                                             │
 ┌───────────────────────┐                   ▼
 │   Generative AI       │       ┌────────────────────────┐
 │ (Google Gemini LLM)   │ <───> │  Hybrid Storage Layer  │
 └───────────────────────┘       │ (Neo4j Graph & FAISS)  │
                                 └────────────────────────┘

```

### 2.1 The Ingestion & Extraction Pipeline

The backend exposes dedicated ingestion endpoints that trigger background or synchronous processing:

* **arXiv Fetcher:** Connects to external APIs to pull paper metadata and abstracts based on keyword queries and max result limits.
* **Semantic Extraction:** Unstructured abstracts are passed to NLP extractors (`extract_triples`), identifying canonical entities (Subject, Object) and their relationships (Predicate) along with confidence scores.

### 2.2 Hybrid Storage Engine

Data is persisted across two specialized systems concurrently:

* **Neo4j Graph:** Entities and relationships are merged idempotently via the `KGMerger`. Full paper text is stored as a node property (`p.text`) to retain source fidelity.
* **Vector Index:** Chunks of the abstract are embedded and stored (e.g., via FAISS) for dense semantic retrieval.

### 2.3 Synthesis & Smart Caching Layer

* **Hybrid RAG:** The `/query` endpoint dynamically flags routes as `HYBRID`, `CYPHER`, or `VECTOR` based on the retrieval strategy.
* **Read-Through Summary Cache:** To minimize LLM token costs and latency, the system queries Neo4j first when a paper summary is requested. If a summary does not exist, Gemini generates a concise 3-bullet point scientific summary, which is then permanently saved to the graph node (`SET p.summary = $summary`).

---

## 3. Graph Database Schema

The graph ontology is engineered to map the relationships inherent in scientific and technical literature.

### 🟢 Entity Nodes (`:Entity`)

Entities are dynamically styled in the UI using a hushed pastel palette based on their ontological category.

* **`name`** *(String)*: The canonical identifier or title.
* **`type`** *(String)*: Categorical label mapped to UI clusters (*Paper*, *Method*, *Model*, *Metric*, *Author*, *Concept*, *Organisation*).
* **`mention_count`** *(Integer)*: Frequency accumulator dictating the physical radius of the node in the force simulation.
* **`text`** *(String)*: The raw document payload or abstract.
* **`summary`** *(String)*: The cached 3-bullet point output generated by the Gemini model.

### 🔗 Semantic Edges (`:RELATION`)

Directed edges mapping dependencies, architectures, or citations.

* **`type`** *(String)*: The extracted predicate (e.g., `OPTIMIZES`, `INTRODUCES`, `USES_METHOD`).
* **`confidence`** *(Float)*: NLP extraction certainty metric.
* **`sources`** *(Array)*: Lineage pointers referencing originating document IDs to ensure traceability.

---

## 4. Comprehensive API Reference

The backend is built on **FastAPI** using Pydantic models for strict payload validation.

### Document Ingestion

**`POST /ingest/arxiv`**
Fetches literature dynamically from arXiv and builds the graph.

```json
// Request
{
  "arxiv_query": "Model Distillation BERT",
  "max_results": 5
}
// Response
{
  "documents_processed": 5,
  "triples_extracted": 142
}

```

**`POST /ingest/text`**
Manually injects raw notes, research snippets, or proprietary text.

```json
// Request
{
  "title": "Quantization Study",
  "text": "We applied INT8 quantization to...",
  "doc_id": "optional_custom_id"
}

```

### Retrieval & Querying

**`POST /query`**
The primary agent workspace endpoint. Returns the synthesized AI answer and optional retrieval context.

```json
// Request
{
  "question": "What methods are used for transformer optimization?",
  "include_context": true
}
// Response
{
  "question": "What methods are used...",
  "answer": "Based on the retrieved graph...",
  "entity_seeds": ["transformer", "optimization"],
  "graph_triple_count": 12,
  "vector_result_count": 4,
  "route": "HYBRID",
  "model": "gemini-1.5-pro",
  "context": { ... }
}

```

### Graph Interactions & UI Support

* **`GET /api/paper/summary/{paper_title}`**: Retrieves the cached Gemini summary for a paper, or generates and stores it if missing.
* **`GET /stats`**: Returns database health metrics (`entities`, `relations`, `papers`, `vector_index_size`).
* **`GET /api/ingestion/history`**: Returns the 20 most recently processed documents and their text lengths.
* **`GET /graph?limit=250`**: Yields highly optimized node and link arrays specifically formatted for D3.js consumption.

---

## 5. Frontend Engineering

The client (`index.html`) is a standalone, high-performance vanilla JavaScript application.

* **Bento Grid Architecture:** The UI employs a CSS Grid layout (`.workbench-grid`) with modular "Bento" cards (`.bento-card`) for statistics, ingestion forms, and chat streams, styled with a hushed light palette (`#fafafa` backgrounds, `#18181b` text).
* **D3.js Force Simulation:** The canvas utilizes `d3-force` with multiple forces acting simultaneously:
* `forceManyBody()` ensures nodes repel each other to prevent clustering.
* Radial positioning forces nodes toward category-specific gravity centers.
* `getNodeRadius` dynamically scales node sizes based on `mention_count`.


* **Interactive Inspector Drawer:** Clicking a node halts the simulation slightly and slides open `.inspector-drawer`, revealing the AI summary, entity badges, and external Scholar/arXiv links.

---

## 6. Local Setup & Deployment

### Prerequisites

* Python 3.10+
* Neo4j Database (Local Desktop or AuraDB cloud instance)
* Google Gemini API Key

### Installation

1. **Clone the Repository**
```bash
git clone [https://github.com/your-username/research-workbench.git](https://github.com/your-username/research-workbench.git)
cd research-workbench

```


2. **Virtual Environment & Dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install fastapi uvicorn pydantic google-generativeai neo4j sentence-transformers tenacity httpx

```


3. **Environment Configuration**
Create a `config.py` or `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password
LLM_MODEL=gemini-1.5-pro

```


4. **Launch the FastAPI Server**
```bash
uvicorn api.main:app --reload --port 8000

```


5. **Launch the UI**
Open `index.html` in any modern web browser. Enter `http://localhost:8000` into the API URL input field and click **Ping** to establish the connection.


