# Setup

## Prerequisites
* Docker and Docker Compose
* Python 3.8+

## Installation & Configuration

1.  **Clone the repository and configure environment variables:**
    ```bash
    cp .env.example .env
    ```
    *Add your `GEMINI_API_KEY` to the `.env` file. You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).*

2.  **Start the Neo4j database:**
    ```bash
    docker-compose up -d
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Start the FastAPI server:**
    ```bash
    uvicorn api.main:app --reload
    ```

5.  **Open the dashboard:**
    ```bash
    open frontend/index.html
    ```

---

## Project Layout

```text
research_agent/
├── ingestion/
│   ├── fetchers.py        # arXiv, RSS, PDF fetchers
│   ├── extractors.py      # Triple extraction via Gemini / Rebel / GLiNER
│   └── scheduler.py       # Prefect flows (runs on interval)
├── graph/
│   ├── merger.py          # Entity resolution, fuzzy deduplication, upsert logic
│   ├── schema.cypher      # Database constraints and indexes
│   └── neo4j_client.py
├── rag/
│   ├── router.py          # Intent routing (cypher / vector / hybrid)
│   ├── retriever.py       # Graph traversal and FAISS vector search
│   └── synthesiser.py     # Grounded answer generation
├── api/
│   └── main.py            # FastAPI endpoints
├── frontend/
│   └── index.html         # User dashboard (query, ingest, stats)
└── docker-compose.yml

```

---

## Key Ideas Worth Stealing

* **Fuzzy Entity Resolution:** Before inserting a new node, the `merger.py` module embeds the entity's name and checks cosine similarity against existing nodes of the same type. If the similarity is above a configurable threshold (default `0.92`), it merges them rather than creating a duplicate (e.g., "GPT-4" and "GPT4" become a single node).
* **Confidence Averaging:** Every extracted triple tracks how many sources confirmed it and averages the confidence scores across them. A relationship seen in 12 different papers with consistent confidence is ranked higher than a one-off mention.
* **Pluggable Triple Extraction Backends:** You can swap between `llm` (highest quality), `rebel` (fast, local), or `gliner` (NER-based, no GPU required) using a single environment variable (`TRIPLE_EXTRACTION_METHOD`). This is incredibly useful for managing API costs or latency constraints.

---

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/query` | Ask a question, get a cited answer |
| `POST` | `/ingest/arxiv` | Fetch and ingest arXiv papers |
| `POST` | `/ingest/text` | Ingest raw text directly |
| `GET` | `/stats` | Get entity, relation, paper, and vector counts |
| `GET` | `/health` | Liveness check |

---

## Environment Variables

Configure these in your `.env` file:

```env
GEMINI_API_KEY=...
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password
RESEARCH_TOPICS=large language models,RAG,knowledge graphs
FETCH_INTERVAL_HOURS=24
MAX_PAPERS_PER_TOPIC=20
TRIPLE_EXTRACTION_METHOD=llm   # Options: llm | rebel | gliner
ENTITY_MERGE_THRESHOLD=0.92
GRAPH_HOP_LIMIT=2
VECTOR_TOP_K=5

```

```

```
