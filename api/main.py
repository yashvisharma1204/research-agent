"""
api/main.py
FastAPI application exposing:
  POST /query          — ask a question against the KG
  POST /ingest/arxiv   — fetch & ingest arXiv papers by query
  POST /ingest/text    — ingest raw text on demand
  GET  /stats          — graph statistics
  GET  /health         — liveness probe
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph.merger import KGMerger
from graph.neo4j_client import get_driver
from ingestion.extractors import extract_triples
from ingestion.fetchers import fetch_arxiv
from rag.retriever import Retriever
from rag.synthesiser import Synthesiser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App state ─────────────────────────────────────────────────────────────────

retriever: Retriever | None = None
synthesiser: Synthesiser | None = None
merger: KGMerger | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, synthesiser, merger
    driver = get_driver()
    retriever = Retriever(driver)
    synthesiser = Synthesiser()
    merger = KGMerger(driver)
    logger.info("Research agent ready")
    yield
    driver.close()


app = FastAPI(
    title="Self-Updating Research Agent",
    description="KG + RAG research assistant with live graph updates (Gemini-powered)",
    version="1.1.0",
    lifespan=lifespan,
)


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    include_context: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    entity_seeds: list[str]
    graph_triple_count: int
    vector_result_count: int
    route: str
    model: str
    context: dict | None = None


class IngestURLRequest(BaseModel):
    arxiv_query: str
    max_results: int = 5


class IngestTextRequest(BaseModel):
    title: str
    text: str
    doc_id: str = ""


class IngestResponse(BaseModel):
    documents_processed: int
    triples_extracted: int


class StatsResponse(BaseModel):
    entities: int
    relations: int
    papers: int
    vector_index_size: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not retriever or not synthesiser:
        raise HTTPException(503, "Agent not initialised")

    context = retriever.retrieve(req.question)
    answer_obj = synthesiser.answer(req.question, context)

    response = QueryResponse(
        question=answer_obj.question,
        answer=answer_obj.answer,
        entity_seeds=answer_obj.entity_seeds,
        graph_triple_count=answer_obj.graph_triple_count,
        vector_result_count=answer_obj.vector_result_count,
        route=context.route,
        model=answer_obj.model,
    )

    if req.include_context:
        response.context = {
            "graph_triples": [
                {"s": t.subject, "p": t.predicate, "o": t.obj, "confidence": t.confidence}
                for t in context.graph_triples
            ],
            "vector_results": [
                {"text": r.text[:200], "score": r.score, "doc_id": r.doc_id}
                for r in context.vector_results
            ],
        }

    return response


@app.post("/ingest/arxiv", response_model=IngestResponse)
async def ingest_arxiv_endpoint(req: IngestURLRequest):
    if not merger or not retriever:
        raise HTTPException(503, "Agent not initialised")

    docs = fetch_arxiv(req.arxiv_query, max_results=req.max_results)
    total_triples = 0

    for doc in docs:
        text = f"{doc['title']}. {doc['abstract']}"
        triples = extract_triples(text, source_id=doc["id"])

        merger.upsert_entity(doc["title"], "Paper", {"url": doc.get("url", ""), "source": "arxiv"})
        for t in triples:
            merger.upsert_triple(t.subject, t.predicate, t.obj, doc["id"], t.confidence)

        retriever.index_document(doc)
        total_triples += len(triples)

    return IngestResponse(documents_processed=len(docs), triples_extracted=total_triples)


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text_endpoint(req: IngestTextRequest):
    if not merger or not retriever:
        raise HTTPException(503, "Agent not initialised")

    doc_id = req.doc_id or req.title
    triples = extract_triples(req.text, source_id=doc_id)

    merger.upsert_entity(req.title, "Document", {})
    for t in triples:
        merger.upsert_triple(t.subject, t.predicate, t.obj, doc_id, t.confidence)

    retriever.index_document({"id": doc_id, "title": req.title, "abstract": req.text[:2000]})

    return IngestResponse(documents_processed=1, triples_extracted=len(triples))


@app.get("/stats", response_model=StatsResponse)
async def stats():
    if not merger or not retriever:
        raise HTTPException(503, "Agent not initialised")

    graph_stats = merger.get_stats()
    return StatsResponse(
        entities=graph_stats["entities"],
        relations=graph_stats["relations"],
        papers=graph_stats["papers"],
        vector_index_size=retriever.vector._index.ntotal,
    )


# ── CORS middleware (fixes browser "Failed to fetch") ─────────────────────────
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Graph endpoint ────────────────────────────────────────────────────────────

class GraphResponse(BaseModel):
    nodes: list[dict]
    links: list[dict]

@app.get("/graph", response_model=GraphResponse)
async def get_graph(limit: int = 120):
    """
    Returns nodes and edges for the D3/Vis.js visualiser.
    Directly extracts properties to avoid Neo4j Path serialization errors.
    """
    if not merger:
        raise HTTPException(503, "Agent not initialised")

    # We explicitly ask for properties instead of a Path object (p)
    cypher = f"""
    MATCH (s:Entity)-[r:RELATION]->(o:Entity)
    RETURN 
        s.name AS s_name, s.type AS s_type, s.mention_count AS s_mention,
        o.name AS e_name, o.type AS e_type, o.mention_count AS e_mention,
        r.type AS r_type, r.confidence AS r_confidence, r.mention_count AS r_mention, r.sources AS r_sources
    LIMIT {limit}
    """

    nodes_map: dict[str, dict] = {}
    links: list[dict] = []

    def neo_int(val) -> int:
        """Convert Neo4j {{ low, high }} integer dict to Python int."""
        if isinstance(val, dict) and "low" in val:
            return val["low"] + val.get("high", 0) * (2 ** 32)
        return int(val) if val is not None else 0

    with get_driver().session() as session:
        records = session.run(cypher).data()

    for rec in records:
        # Use the entity name as the unique ID for the graph
        s_id = rec.get("s_name") or "?"
        e_id = rec.get("e_name") or "?"

        if s_id not in nodes_map:
            nodes_map[s_id] = {
                "id": s_id,
                "name": s_id,
                "type": rec.get("s_type", "unknown"),
                "mention_count": neo_int(rec.get("s_mention", 1)),
            }
        if e_id not in nodes_map:
            nodes_map[e_id] = {
                "id": e_id,
                "name": e_id,
                "type": rec.get("e_type", "unknown"),
                "mention_count": neo_int(rec.get("e_mention", 1)),
            }

        links.append({
            "source": s_id,
            "target": e_id,
            "predicate": rec.get("r_type", "related_to"),
            "confidence": float(rec.get("r_confidence", 1.0) or 1.0),
            "mention_count": neo_int(rec.get("r_mention", 1)),
            "sources": rec.get("r_sources", []),
        })

    return GraphResponse(nodes=list(nodes_map.values()), links=links)
