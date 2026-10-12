"""
api/main.py
FastAPI application exposing:
  POST /query                  — ask a question against the KG
  POST /ingest/arxiv           — fetch & ingest arXiv papers by query
  POST /ingest/text            — ingest raw text on demand
  GET  /stats                  — graph statistics
  GET  /health                 — liveness probe
  GET  /graph                  — D3/Vis.js visualiser endpoint
  GET  /api/ingestion/history  — recent ingestion history
  GET  /api/paper/summary/{paper_title} — real AI summary generator
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.genai import types

import fitz  # PyMuPDF — PDF text extraction

from api.graph_events import hub

from config import cfg
from graph.merger import KGMerger
from graph.neo4j_client import get_driver
from ingestion.extractors import extract_triples, extract_triples_and_results, extract_triples_results_contributions
from ingestion.fetchers import fetch_foundational
from rag.retriever import Retriever
from rag.synthesiser import Synthesiser
from utils.gemini import get_client
import os
import secrets
from fastapi.security import HTTPBasic, HTTPBasicCredentials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security = HTTPBasic()

def verify_ingestion_password(credentials: HTTPBasicCredentials = Depends(security)):
    expected_password = os.getenv("INGESTION_PASSWORD")

    # Fail closed: if no password is configured, ingestion is disabled rather
    # than protected by a well-known hardcoded fallback secret.
    if not expected_password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion is disabled: INGESTION_PASSWORD is not configured.",
        )

    expected_password_bytes = expected_password.strip('"').encode("utf8")
    supplied_password_bytes = (credentials.password or "").encode("utf8")

    # Password-only auth: the HTTP Basic username field is ignored; only the
    # password is validated, using a constant-time comparison.
    if not secrets.compare_digest(supplied_password_bytes, expected_password_bytes):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access restricted. Invalid ingestion password.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

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
    # Capture the running loop so the (sync) ingestion path can broadcast live
    # graph deltas to WebSocket clients via hub.publish_threadsafe().
    hub.bind_loop(asyncio.get_running_loop())
    logger.info("Research agent ready")
    yield
    driver.close()
    # Clear the cached driver so a subsequent startup (e.g. --reload) creates a
    # fresh, open driver instead of returning this now-closed instance.
    get_driver.cache_clear()


app = FastAPI(
    title="Self-Updating Research Agent",
    description="KG + RAG research assistant with live graph updates (Gemini-powered)",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS origins are configurable via env (comma-separated). Browsers reject the
# combination of wildcard origin + credentials, so only enable credentials when
# explicit origins are provided.
_cors_origins = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()
]
_allow_credentials = _cors_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
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
    is_faithful: bool         
    critique_notes: str        
    context: dict | None = None

class IngestURLRequest(BaseModel):
    arxiv_query: str
    max_results: int = 5


class IngestTextRequest(BaseModel):
    title: str
    text: str
    doc_id: str = ""
    published_date: str = "Unknown"


class IngestResponse(BaseModel):
    documents_processed: int
    triples_extracted: int


class StatsResponse(BaseModel):
    entities: int
    relations: int
    papers: int
    vector_index_size: int
    authors: int = 0
    results: int = 0


class GraphResponse(BaseModel):
    nodes: list[dict]
    links: list[dict]


# ── Helper Function: Live graph deltas over WebSocket ─────────────────────────

def _n(name: str, ntype: str) -> dict:
    """Shape a node exactly like the /graph endpoint so the frontend can reuse
    its existing merge/render path verbatim."""
    return {"id": name, "name": name, "type": ntype or "Unknown", "mention_count": 1}


def build_ingest_delta(doc: dict, triples: list, results: list) -> dict:
    """Build a `{nodes, links}` delta for one freshly-ingested document.

    Derived from the in-memory extraction output (no extra Neo4j round-trip) and
    deduplicated by id/edge so the payload is compact. Node/link shapes mirror
    GET /graph so the client applies them through one code path. The frontend is
    responsible for discarding ids it already has — deltas are advisory, not
    authoritative, which keeps this resilient to fuzzy entity-resolution renames.
    """
    nodes: dict[str, dict] = {}
    links: list[dict] = []
    seen_links: set[tuple] = set()

    # Paper node (if this was a paper ingest) plus its author/citation subgraph.
    title = doc.get("title") if doc else None
    if title:
        nodes[title] = _n(title, "Paper")
        for author in (doc.get("authors") or []):
            author = (author or "").strip()
            if not author:
                continue
            nodes.setdefault(author, _n(author, "Author"))
            key = (author, title, "authored")
            if key not in seen_links:
                seen_links.add(key)
                links.append({"source": author, "target": title,
                              "predicate": "authored", "confidence": 1.0,
                              "mention_count": 1, "sources": []})

    # Extracted semantic triples (:RELATION edges).
    for t in triples or []:
        nodes.setdefault(t.subject, _n(t.subject, t.subject_type))
        nodes.setdefault(t.obj, _n(t.obj, t.obj_type))
        key = (t.subject, t.obj, t.predicate)
        if key not in seen_links:
            seen_links.add(key)
            links.append({"source": t.subject, "target": t.obj,
                          "predicate": t.predicate,
                          "confidence": float(getattr(t, "confidence", 1.0) or 1.0),
                          "mention_count": 1, "sources": []})

    # Reified benchmark results: (:Model)->(:Result)->(:Dataset)/(:Metric).
    for r in results or []:
        model = (getattr(r, "model", "") or "").strip()
        dataset = (getattr(r, "dataset", "") or "").strip()
        metric = (getattr(r, "metric", "") or "").strip()
        if not (model and dataset and metric):
            continue
        rid = f"{model}::{dataset}::{metric}"
        nodes.setdefault(model, _n(model, "Model"))
        nodes.setdefault(dataset, _n(dataset, "Dataset"))
        nodes.setdefault(metric, _n(metric, "Metric"))
        nodes.setdefault(rid, {"id": rid, "name": getattr(r, "value", "") or "Result",
                               "type": "Result", "mention_count": 1})
        for src, dst, pred in ((model, rid, "achieves"), (rid, dataset, "on"),
                               (rid, metric, "measured_by")):
            key = (src, dst, pred)
            if key not in seen_links:
                seen_links.add(key)
                links.append({"source": src, "target": dst, "predicate": pred,
                              "confidence": 1.0, "mention_count": 1, "sources": []})

    return {"type": "graph_delta", "nodes": list(nodes.values()), "links": links}


def broadcast_delta(doc: dict, triples: list, results: list) -> None:
    """Best-effort publish of a document's delta to live WS clients.

    Never raises into the ingestion path: a visualisation feature must not be
    able to fail a MERGE. Skips the work entirely when nobody is listening.
    """
    try:
        if hub.client_count == 0:
            return
        delta = build_ingest_delta(doc, triples, results)
        if delta["nodes"] or delta["links"]:
            hub.publish_threadsafe(delta)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Graph delta broadcast skipped: %s", exc)


# ── Helper Function: Extract text from an uploaded PDF (PyMuPDF) ──────────────

# Guard rails for PDF ingestion.
_MAX_PDF_BYTES = 25 * 1024 * 1024          # 25 MB upload cap
_MAX_PDF_EXTRACT_CHARS = 200_000           # cap extracted text to bound LLM cost


def extract_pdf_text(data: bytes) -> tuple[str, str]:
    """Extract concatenated page text and a best-effort title from PDF *bytes*.

    Returns (text, title). Title falls back to '' when PDF metadata has none;
    the caller supplies a filename-based default. Raises ValueError on an
    unreadable / non-PDF payload so the route can return a clean 400.
    """
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not open PDF: {exc}") from exc

    try:
        parts: list[str] = []
        total = 0
        for page in doc:
            parts.append(page.get_text("text"))
            total += len(parts[-1])
            if total >= _MAX_PDF_EXTRACT_CHARS:
                break
        text = "\n".join(parts).strip()[:_MAX_PDF_EXTRACT_CHARS]
        meta_title = (doc.metadata or {}).get("title", "") or ""
    finally:
        doc.close()

    return text, meta_title.strip()


# ── Helper Function: Generate Scientific Summary ──────────────────────────────

def generate_paper_summary(paper_title: str, text: str, relations: list) -> str:
    """Uses Gemini to generate a strict 3-bullet point scientific summary with no preamble."""
    prompt = f"""Provide a concise 3-bullet point scientific summary of this paper based on its abstract text and connected knowledge graph relations. 
    
    CRITICAL INSTRUCTION: Output ONLY the 3 bullet points. Do not include any conversational intro, title, or outro text.

    Paper Title: {paper_title}
    Abstract Text: {text[:3000]}
    Connected Graph Relations: {relations[:15]}
    """
    
    gemini_client = get_client()
    response = gemini_client.models.generate_content(
        model=cfg.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=512),
    )
    return (response.text or "").strip()

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/graph")
async def ws_graph(websocket: WebSocket):
    """Live graph delta stream.

    On connect the server sends a `{"type": "hello"}` frame; thereafter it
    pushes `{"type": "graph_delta", "nodes": [...], "links": [...]}` frames each
    time a document is ingested. The client reconciles deltas against its own
    node set, so no initial snapshot is sent here — the client already loaded
    one via GET /graph. Inbound messages are ignored (one-way stream) but we
    keep reading so the socket close handshake is detected promptly.
    """
    await hub.connect(websocket)
    try:
        await websocket.send_json({"type": "hello", "message": "graph stream connected"})
        while True:
            # We don't expect client messages; this await returns when the
            # client disconnects, raising WebSocketDisconnect.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("Graph WS error: %s", exc)
    finally:
        await hub.disconnect(websocket)


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
        is_faithful=answer_obj.is_faithful,         # <--- Map it here
        critique_notes=answer_obj.critique_notes,   # <--- Map it here
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

@app.post("/ingest/arxiv", response_model=IngestResponse, dependencies=[Depends(verify_ingestion_password)])
async def ingest_arxiv_endpoint(req: IngestURLRequest):
    if not merger or not retriever:
        raise HTTPException(503, "Agent not initialised")

    docs = fetch_foundational(req.arxiv_query, max_results=req.max_results)
    total_triples = 0

    for doc in docs:
        text = f"{doc['title']}. {doc['abstract']}"
        
        # Safely get the published date
        pub_date = doc.get("published_date") or "Unknown"

        # Pass the date into the extractor
        triples, results, contributions = extract_triples_results_contributions(text, source_id=doc["id"], published_date=pub_date, paper_title=doc["title"])

        # Store paper as a first-class :Paper node (metadata + author/citation subgraph)
        merger.upsert_paper(doc)

        # paper_name anchors the paper node to the entities extracted from its
        # own abstract (:Paper)-[:MENTIONS]->(:Entity), so it is never an orphan.
        merger.upsert_document_triples(triples, doc["id"], paper_name=doc["title"])
        merger.upsert_results(results, doc["id"])
        # Reified contributions linked back to the :Paper node (keyed on title).
        merger.upsert_contributions(contributions, doc["id"], paper_name=doc["title"])

        retriever.index_document(doc)
        total_triples += len(triples)

        # Push the newly-created subgraph to any live WebSocket viewers so the
        # D3 canvas animates this document in as it lands.
        broadcast_delta(doc, triples, results)

    return IngestResponse(
        documents_processed=len(docs),
        triples_extracted=total_triples
    )


@app.post("/ingest/text", response_model=IngestResponse, dependencies=[Depends(verify_ingestion_password)])
async def ingest_text_endpoint(req: IngestTextRequest):
    if not merger or not retriever:
        raise HTTPException(503, "Agent not initialised")

    doc_id = req.doc_id or req.title
    
    # Pass the date into the extractor
    triples, results, contributions = extract_triples_results_contributions(req.text, source_id=doc_id, published_date=req.published_date, paper_title=req.title)

    # Persist the date in Neo4j
    merger.upsert_entity(req.title, "Document", {
        "text": req.text,
        "published_date": req.published_date,
    })
    
    merger.upsert_document_triples(triples, doc_id, paper_name=req.title)
    merger.upsert_results(results, doc_id)
    # Reified contributions linked back to the :Document node (keyed on title).
    merger.upsert_contributions(contributions, doc_id, paper_name=req.title)

    # Pass the date into the vector database
    retriever.index_document({
        "id": doc_id, 
        "title": req.title, 
        "abstract": req.text[:2000],
        "published_date": req.published_date
    })

    # Live-stream this document's subgraph to connected WS viewers. The text
    # path has no author/citation metadata, so pass a title-only doc.
    broadcast_delta({"title": req.title}, triples, results)

    return IngestResponse(documents_processed=1, triples_extracted=len(triples))


@app.post("/ingest/pdf", response_model=IngestResponse, dependencies=[Depends(verify_ingestion_password)])
async def ingest_pdf_endpoint(
    file: UploadFile = File(...),
    title: str = Form(""),
    published_date: str = Form("Unknown"),
):
    """Ingest a full-text PDF (goes beyond abstracts).

    Accepts multipart/form-data: a `file` (the PDF) plus optional `title` and
    `published_date` fields. Text is extracted with PyMuPDF, then run through the
    exact same pipeline as /ingest/text (triple + result extraction, idempotent
    KG merge, chunk vector indexing, live WS delta). Protected by the ingestion
    password like the other /ingest/* routes.
    """
    if not merger or not retriever:
        raise HTTPException(503, "Agent not initialised")

    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf") and (file.content_type or "") != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")
    if len(data) > _MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 25 MB upload limit.")

    try:
        text, meta_title = extract_pdf_text(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not text:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in the PDF (it may be a scanned image).",
        )

    # Title precedence: explicit form field > PDF metadata title > filename stem.
    resolved_title = (title.strip() or meta_title or filename.rsplit("/", 1)[-1][:-4]).strip()
    if not resolved_title:
        resolved_title = "Untitled PDF"
    doc_id = resolved_title

    triples, results = extract_triples_and_results(
        text, source_id=doc_id, published_date=published_date, paper_title=resolved_title
    )

    merger.upsert_entity(resolved_title, "Document", {
        "text": text,
        "published_date": published_date,
    })
    merger.upsert_document_triples(triples, doc_id, paper_name=resolved_title)
    merger.upsert_results(results, doc_id)

    retriever.index_document({
        "id": doc_id,
        "title": resolved_title,
        "text": text,
        "published_date": published_date,
    })

    broadcast_delta({"title": resolved_title}, triples, results)

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
        vector_index_size=graph_stats.get("vectors", 0),
        authors=graph_stats.get("authors", 0),
        results=graph_stats.get("results", 0),
    )


@app.get("/api/paper/summary/{paper_title}")
async def get_paper_summary(paper_title: str):
    """Fetches or generates a stored summary, saving it to Neo4j on the first request."""
    cypher_fetch = """
    MATCH (p:Entity {name: $title})
    OPTIONAL MATCH (p)-[r]->(o:Entity)
    RETURN p.text AS text, p.summary AS summary, collect({pred: type(r), obj: o.name}) AS relations
    """
    
    try:
        with get_driver().session() as session:
            record = session.run(cypher_fetch, title=paper_title).single()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")
        
    if not record or not record.get("text"):
        raise HTTPException(status_code=404, detail="Paper text or abstract not found in graph storage.")
        
    paper_text = record["text"]
    stored_summary = record.get("summary")
    
    # Return cached summary if it already exists in Neo4j
    if stored_summary:
        return {
            "title": paper_title,
            "summary": stored_summary,
            "source_text_snippet": paper_text[:400] + "..."
        }
    
    # Otherwise, generate it via Gemini
    relations = record["relations"]
    try:
        summary = generate_paper_summary(paper_title, paper_text, relations)
        
        # Store the generated summary directly back into the Neo4j Paper node
        cypher_save = """
        MATCH (p:Entity {name: $title})
        SET p.summary = $summary
        """
        with get_driver().session() as session:
            session.run(cypher_save, title=paper_title, summary=summary)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation error: {str(e)}")
    
    return {
        "title": paper_title,
        "summary": summary,
        "source_text_snippet": paper_text[:400] + "..."
    }

@app.get("/api/ingestion/history")
async def get_ingestion_history():
    """Retrieves history of ingested papers/documents from the graph."""
    cypher = """
    MATCH (e:Entity)
    WHERE e.text IS NOT NULL
    RETURN DISTINCT
        e.name AS title,
        e.type AS type,
        size(coalesce(e.text, '')) AS text_length,
        e.published_date AS published_date,
        coalesce(e.created, e.updated, 0) AS created
    ORDER BY created DESC, title ASC
    LIMIT 20
    """
    try:
        with get_driver().session() as session:
            records = session.run(cypher).data()
        return {"history": records}
    except Exception as exc:
        logger.error("Failed to fetch ingestion history: %s", exc)
        return {"history": []}

@app.get("/graph", response_model=GraphResponse)
async def get_graph(limit: int = 120):
    """Returns graph nodes and edges for visualizer UI."""
    if not merger:
        raise HTTPException(503, "Agent not initialised")

    # 1) Extracted semantic triples (generic :RELATION edges).
    #    A :Contribution node's `name` is an internal id (prefixed with the
    #    source doc URL). For display we surface the owning paper's TITLE (via
    #    the makes_claim edge) so the node reads as the paper, while remaining a
    #    first-class :Contribution. The paper url is passed through for the drawer.
    relation_cypher = """
    MATCH (s:Entity)-[r:RELATION]->(o:Entity)
    OPTIONAL MATCH (sp:Paper)-[:RELATION {type:'makes_claim'}]->(s)
      WHERE s:Contribution
    OPTIONAL MATCH (op:Paper)-[:RELATION {type:'makes_claim'}]->(o)
      WHERE o:Contribution
    RETURN
        s.name AS s_name, s.type AS s_type, s.mention_count AS s_mention,
        coalesce(sp.name, s.name) AS s_label, sp.url AS s_url, s.statement AS s_statement,
        o.name AS e_name, o.type AS e_type, o.mention_count AS e_mention,
        coalesce(op.name, o.name) AS e_label, op.url AS e_url, o.statement AS e_statement,
        r.type AS r_type, r.confidence AS r_confidence, r.mention_count AS r_mention, r.sources AS r_sources
    LIMIT $limit
    """

    # 2) Structural + reified edges introduced by the ontology redesign.
    #    Each row carries typed endpoints so authors, papers, and reified
    #    result subgraphs surface in the visualiser (they never use :RELATION).
    #    :Result nodes are NOT :Entity-labelled, so they are coalesced with a
    #    literal "Result" type for palette colouring on the frontend.
    structural_cypher = """
    MATCH (a)-[r:AUTHORED|CITES|ACHIEVES|ON|MEASURED_BY|MENTIONS]->(b)
    OPTIONAL MATCH (ap:Paper)-[:RELATION {type:'makes_claim'}]->(a)
      WHERE a:Contribution
    OPTIONAL MATCH (bp:Paper)-[:RELATION {type:'makes_claim'}]->(b)
      WHERE b:Contribution
    RETURN
        a.name AS s_name,
        coalesce(a.type, head([l IN labels(a) WHERE l <> 'Entity'])) AS s_type,
        a.mention_count AS s_mention,
        coalesce(ap.name, a.name) AS s_label, coalesce(ap.url, a.url) AS s_url, a.statement AS s_statement,
        b.name AS e_name,
        coalesce(b.type, head([l IN labels(b) WHERE l <> 'Entity'])) AS e_type,
        b.mention_count AS e_mention,
        coalesce(bp.name, b.name) AS e_label, coalesce(bp.url, b.url) AS e_url, b.statement AS e_statement,
        type(r) AS r_type
    LIMIT $limit
    """

    # 3) Any remaining nodes with no rendered edge, so the visualiser total
    #    lines up with /stats instead of silently dropping isolated nodes.
    isolated_cypher = """
    MATCH (n)
    WHERE (n:Entity OR n:Result)
    OPTIONAL MATCH (np:Paper)-[:RELATION {type:'makes_claim'}]->(n)
      WHERE n:Contribution
    RETURN
        n.name AS name,
        coalesce(n.type, head([l IN labels(n) WHERE l <> 'Entity'])) AS type,
        n.mention_count AS mention,
        coalesce(np.name, n.name) AS label, coalesce(np.url, n.url) AS url, n.statement AS statement
    LIMIT $node_limit
    """

    nodes_map: dict[str, dict] = {}
    links: list[dict] = []

    def neo_int(val) -> int:
        if isinstance(val, dict) and "low" in val:
            return val["low"] + val.get("high", 0) * (2 ** 32)
        return int(val) if val is not None else 0

    def add_node(node_id, node_type, mention, label=None, url=None, statement=None):
        if not node_id:
            return
        if node_id not in nodes_map:
            display = (label or node_id)
            node = {
                "id": node_id,
                "name": display,
                "type": node_type or "Unknown",
                "mention_count": neo_int(mention if mention is not None else 1),
            }
            # Expose the source paper URL (used by the inspector drawer's
            # "View paper" link), e.g. for Contribution nodes pointing back to
            # the owning paper.
            if url:
                node["url"] = url
            # The contribution's own one-line claim, shown in the drawer body
            # (the node label itself is the owning paper's title).
            if statement:
                node["statement"] = statement
            nodes_map[node_id] = node

    with get_driver().session() as session:
        relation_records = session.run(relation_cypher, limit=limit).data()
        structural_records = session.run(structural_cypher, limit=limit).data()
        # Allow enough headroom to backfill isolated nodes up to the node budget.
        isolated_records = session.run(isolated_cypher, node_limit=limit * 4).data()

    for rec in relation_records:
        s_id = rec.get("s_name")
        e_id = rec.get("e_name")
        # A relation whose endpoint has no name is a malformed node — skip it
        # rather than collapsing every such edge into one shared "?" hub.
        if not s_id or not e_id:
            continue
        add_node(s_id, rec.get("s_type"), rec.get("s_mention"), rec.get("s_label"), rec.get("s_url"), rec.get("s_statement"))
        add_node(e_id, rec.get("e_type"), rec.get("e_mention"), rec.get("e_label"), rec.get("e_url"), rec.get("e_statement"))
        links.append({
            "source": s_id,
            "target": e_id,
            "predicate": rec.get("r_type", "related_to"),
            "confidence": float(rec.get("r_confidence", 1.0) or 1.0),
            "mention_count": neo_int(rec.get("r_mention", 1)),
            "sources": rec.get("r_sources", []),
        })

    for rec in structural_records:
        s_id = rec.get("s_name")
        e_id = rec.get("e_name")
        if not s_id or not e_id:
            continue
        add_node(s_id, rec.get("s_type"), rec.get("s_mention"), rec.get("s_label"), rec.get("s_url"), rec.get("s_statement"))
        add_node(e_id, rec.get("e_type"), rec.get("e_mention"), rec.get("e_label"), rec.get("e_url"), rec.get("e_statement"))
        links.append({
            "source": s_id,
            "target": e_id,
            "predicate": (rec.get("r_type") or "related_to").lower(),
            "confidence": 1.0,
            "mention_count": 1,
            "sources": [],
        })

    # Backfill isolated nodes (no rendered edge) so counts match /stats.
    for rec in isolated_records:
        add_node(rec.get("name"), rec.get("type"), rec.get("mention"), rec.get("label"), rec.get("url"), rec.get("statement"))

    return GraphResponse(nodes=list(nodes_map.values()), links=links)