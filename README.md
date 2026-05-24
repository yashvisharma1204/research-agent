<div align="center">

# Self Evolving Research Agent

<img width="812" height="541" alt="Self Evolving Research Agent" src="https://github.com/user-attachments/assets/ce20ad31-7171-4716-ba50-3a07ab22df79" />

*A knowledge graph that learns while you sleep.*

</div>

---

Most RAG systems are frozen in time — you feed them documents once and they stay dumb. This one doesn't. Every few hours it reaches out to arXiv and RSS feeds, pulls the latest papers on topics you care about, extracts structured knowledge triples, and weaves them into a live Neo4j graph. Ask it a question tomorrow and it knows things it didn't know today.

## How it thinks

When you ask a question, a router first decides *how* to answer it:

- **Cypher** — for questions about specific entities and their relationships (`"What does GPT-4 outperform?"`)
- **Vector** — for open-ended semantic exploration (`"Recent trends in few-shot learning?"`)
- **Hybrid** — both, fused into one context bundle

A Gemini model then synthesises an answer grounded in graph triples + document excerpts, citing every claim back to a source.

## Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 2.0 Flash (free tier) |
| Knowledge Graph | Neo4j |
| Vector Index | FAISS + `all-MiniLM-L6-v2` |
| Triple Extraction | Gemini LLM / Rebel / GLiNER |
| Ingestion Schedule | Prefect flows |
| API | FastAPI |
| Frontend | Vanilla HTML/CSS/JS |

## Setup

```bash
# 1. Clone and configure
cp .env.example .env
# Add your GEMINI_API_KEY → https://aistudio.google.com/app/apikey

# 2. Start Neo4j
docker-compose up -d

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the API
uvicorn api.main:app --reload

# 5. Open the dashboard
open frontend/index.html
```

## Project layout

```
research_agent/
├── ingestion/
│   ├── fetchers.py        # arXiv, RSS, PDF
│   ├── extractors.py      # Gemini / Rebel / GLiNER triple extraction
│   └── scheduler.py       # Prefect flows (runs on interval)
├── graph/
│   ├── merger.py          # entity resolution + fuzzy dedup + upsert
│   ├── schema.cypher      # constraints and indexes
│   └── neo4j_client.py
├── rag/
│   ├── router.py          # intent → cypher / vector / hybrid
│   ├── retriever.py       # graph traversal + FAISS search
│   └── synthesiser.py     # grounded answer generation
├── api/
│   └── main.py            # FastAPI endpoints
├── frontend/
│   └── index.html         # dashboard (query · ingest · stats)
└── docker-compose.yml
```

## Key ideas worth stealing

**Fuzzy entity resolution** — before inserting a new node, the merger embeds its name and checks cosine similarity against existing nodes of the same type. Above a configurable threshold (default 0.92), it merges rather than duplicates. "GPT-4" and "GPT4" become one node.

**Confidence averaging** — every triple tracks how many sources confirmed it and averages confidence across them. A relationship seen in 12 papers with consistent confidence is ranked above a one-off mention.

**Triple extraction backends** — swap between `llm` (best quality), `rebel` (fast, local), or `gliner` (NER-based, no GPU needed) via a single env variable. Useful when API costs or latency matter.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/query` | Ask a question, get a cited answer |
| `POST` | `/ingest/arxiv` | Fetch and ingest arXiv papers |
| `POST` | `/ingest/text` | Ingest raw text directly |
| `GET` | `/stats` | Entity, relation, paper, vector counts |
| `GET` | `/health` | Liveness check |

## Environment variables

```env
GEMINI_API_KEY=...
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password
RESEARCH_TOPICS=large language models,RAG,knowledge graphs
FETCH_INTERVAL_HOURS=24
MAX_PAPERS_PER_TOPIC=20
TRIPLE_EXTRACTION_METHOD=llm   # llm | rebel | gliner
ENTITY_MERGE_THRESHOLD=0.92
GRAPH_HOP_LIMIT=2
VECTOR_TOP_K=5
```

---

<div align="center">
<sub>Built with Gemini · Neo4j · FAISS · FastAPI · Prefect</sub>
</div>