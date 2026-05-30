"""
api/main.py — full file with learning path endpoint added
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph.merger import KGMerger
from graph.neo4j_client import get_driver
from ingestion.extractors import extract_triples, analyse_paper
from ingestion.fetchers import fetch_foundational
from rag.retriever import Retriever
from rag.synthesiser import Synthesiser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

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

class GraphResponse(BaseModel):
    nodes: list[dict]
    links: list[dict]

class LearningPathRequest(BaseModel):
    topic: str
    max_papers: int = 8

class LearningPathPaper(BaseModel):
    title: str
    year: str
    reason: str
    key_concepts: list[str] = []
    difficulty: str = "intermediate"
    estimated_read_time: str = ""
    url: str = ""

class LearningPathStageOut(BaseModel):
    name: str
    description: str
    papers: list[LearningPathPaper] = []

class LearningPathResponse(BaseModel):
    topic: str
    overview: str
    stages: list[LearningPathStageOut]
    what_you_will_learn: list[str] = []
    total_papers: int


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
            "graph_triples": [{"s":t.subject,"p":t.predicate,"o":t.obj,"confidence":t.confidence} for t in context.graph_triples],
            "vector_results": [{"text":r.text[:200],"score":r.score,"doc_id":r.doc_id} for r in context.vector_results],
        }
    return response


@app.post("/ingest/arxiv", response_model=IngestResponse)
async def ingest_arxiv_endpoint(req: IngestURLRequest):
    if not merger or not retriever:
        raise HTTPException(503, "Agent not initialised")
    docs = fetch_foundational(req.arxiv_query, max_results=req.max_results)
    total_triples = 0
    for doc in docs:
        # Full paper analysis
        paper_node = analyse_paper(doc)
        merger.upsert_paper_node(paper_node)
        # Also store triples with types
        for t in paper_node.triples:
            merger.upsert_entity(t.subject, t.subject_type, {})
            merger.upsert_entity(t.obj, t.obj_type, {})
            merger.upsert_triple(t.subject, t.predicate, t.obj, doc["id"], t.confidence)
        retriever.index_document(doc)
        total_triples += len(paper_node.triples)
    return IngestResponse(documents_processed=len(docs), triples_extracted=total_triples)


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text_endpoint(req: IngestTextRequest):
    if not merger or not retriever:
        raise HTTPException(503, "Agent not initialised")
    doc_id = req.doc_id or req.title
    doc = {"id": doc_id, "title": req.title, "abstract": req.text, "authors": [], "url": ""}
    paper_node = analyse_paper(doc)
    merger.upsert_paper_node(paper_node)
    for t in paper_node.triples:
        merger.upsert_entity(t.subject, t.subject_type, {})
        merger.upsert_entity(t.obj, t.obj_type, {})
        merger.upsert_triple(t.subject, t.predicate, t.obj, doc_id, t.confidence)
    retriever.index_document({"id": doc_id, "title": req.title, "abstract": req.text[:2000]})
    return IngestResponse(documents_processed=1, triples_extracted=len(paper_node.triples))


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


@app.get("/graph", response_model=GraphResponse)
async def get_graph(limit: int = 120):
    if not merger:
        raise HTTPException(503, "Agent not initialised")

    cypher = f"""
    MATCH p = (s)-[r:RELATION]->(o)
    WHERE s:Entity OR s:Paper
    RETURN p LIMIT {limit}
    """

    nodes_map: dict[str, dict] = {}
    links: list[dict] = []

    def neo_int(val) -> int:
        if isinstance(val, dict) and "low" in val:
            return val["low"] + val["high"] * (2**32)
        return int(val) if val is not None else 0

    with get_driver().session() as session:
        records = session.run(cypher).data()

    for rec in records:
        p = rec.get("p", {})
        for seg in p.get("segments", []):
            s_node = seg.get("start", {})
            e_node = seg.get("end", {})
            rel    = seg.get("relationship", {})
            s_props = s_node.get("properties", {})
            e_props = e_node.get("properties", {})
            r_props = rel.get("properties", {})
            s_labels = s_node.get("labels", [])
            e_labels = e_node.get("labels", [])
            s_id = s_node.get("elementId") or str(neo_int(s_node.get("identity", {})))
            e_id = e_node.get("elementId") or str(neo_int(e_node.get("identity", {})))

            if s_id not in nodes_map:
                nodes_map[s_id] = {
                    "id": s_id,
                    "name": s_props.get("name") or s_props.get("title", "?"),
                    "type": s_props.get("type") or ("Paper" if "Paper" in s_labels else "Unknown"),
                    "mention_count": neo_int(s_props.get("mention_count", 1)),
                    "url": s_props.get("url", ""),
                    "year": s_props.get("year", ""),
                    "citations": neo_int(s_props.get("citations", 0)),
                    "summary": s_props.get("summary", ""),
                }
            if e_id not in nodes_map:
                nodes_map[e_id] = {
                    "id": e_id,
                    "name": e_props.get("name") or e_props.get("title", "?"),
                    "type": e_props.get("type") or ("Paper" if "Paper" in e_labels else "Unknown"),
                    "mention_count": neo_int(e_props.get("mention_count", 1)),
                    "url": e_props.get("url", ""),
                    "year": e_props.get("year", ""),
                    "citations": neo_int(e_props.get("citations", 0)),
                    "summary": e_props.get("summary", ""),
                }
            links.append({
                "source": s_id,
                "target": e_id,
                "predicate": r_props.get("type", "related_to"),
                "confidence": float(r_props.get("confidence", 1.0)),
                "mention_count": neo_int(r_props.get("mention_count", 1)),
            })

    return GraphResponse(nodes=list(nodes_map.values()), links=links)


@app.post("/learning-path", response_model=LearningPathResponse)
async def learning_path(req: LearningPathRequest):
    """
    Given a topic:
    1. Fetch foundational papers (curated + citation-sorted)
    2. Run full paper analysis on each (concepts, methods, builds_on)
    3. Store everything in the graph
    4. Ask Gemini to order them into a staged learning path
    """
    if not merger or not retriever or not synthesiser:
        raise HTTPException(503, "Agent not initialised")

    logger.info("Learning path request: topic='%s' max=%d", req.topic, req.max_papers)

    # 1. Fetch papers
    docs = fetch_foundational(req.topic, max_results=req.max_papers)
    if not docs:
        raise HTTPException(404, f"No papers found for topic: {req.topic}")

    # 2. Analyse each paper and store in graph
    enriched_docs = []
    for doc in docs:
        paper_node = analyse_paper(doc)
        merger.upsert_paper_node(paper_node)
        for t in paper_node.triples:
            merger.upsert_entity(t.subject, t.subject_type, {})
            merger.upsert_entity(t.obj, t.obj_type, {})
            merger.upsert_triple(t.subject, t.predicate, t.obj, doc["id"], t.confidence)
        retriever.index_document(doc)

        # Merge analysis results into doc for synthesiser
        enriched = {**doc,
            "concepts":         paper_node.concepts,
            "methods":          paper_node.methods,
            "datasets":         paper_node.datasets,
            "metrics":          paper_node.metrics,
            "key_findings":     paper_node.key_findings,
            "builds_on":        paper_node.builds_on,
            "difficulty":       paper_node.difficulty,
            "one_line_summary": paper_node.one_line_summary,
            "year": paper_node.year,
        }
        enriched_docs.append(enriched)

    # 3. Generate learning path
    path = synthesiser.generate_learning_path(req.topic, enriched_docs)

    # 4. Serialise to response
    stages_out = []
    for stage in path.stages:
        papers_out = []
        for p in stage.papers:
            if isinstance(p, dict):
                papers_out.append(LearningPathPaper(
                    title=p.get("title",""),
                    year=str(p.get("year",""))[:4],
                    reason=p.get("reason",""),
                    key_concepts=p.get("key_concepts",[]),
                    difficulty=p.get("difficulty","intermediate"),
                    estimated_read_time=p.get("estimated_read_time",""),
                    url=p.get("url",""),
                ))
        stages_out.append(LearningPathStageOut(
            name=stage.name,
            description=stage.description,
            papers=papers_out,
        ))

    return LearningPathResponse(
        topic=path.topic,
        overview=path.overview,
        stages=stages_out,
        what_you_will_learn=path.what_you_will_learn,
        total_papers=path.total_papers,
    )