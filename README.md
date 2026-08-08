<div align="center">

# Self-Evolving Research Agent

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Database-Neo4j_Aura-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![FAISS](https://img.shields.io/badge/Vector_Index-FAISS-7A8494?style=flat-square&logo=cpu&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.0_Flash-228B22?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![ONNX](https://img.shields.io/badge/Inference-ONNX_Runtime-005CBB?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)

<img width="812" height="541" alt="Self Evolving Research Agent" src="https://github.com/user-attachments/assets/ce20ad31-7171-4716-ba50-3a07ab22df79" />

*A knowledge graph that learns while you sleep.*

</div>

**[Home](#)** | [Usage](USAGE.md) | [Architecture](ARCHITECTURE.md) | [Contributing]()

---

Most Retrieval-Augmented Generation (RAG) systems operate on static, point-in-time document embeddings. The **Self-Evolving Research Agent** implements a continuous, hybrid RAG architecture. By orchestrating automated ingestion pipelines, structured semantic extraction, and a dynamic graph database, the system continuously weaves incoming unstructured data into a live **Neo4j** knowledge graph. The result is an intelligent backend capable of deterministic multi-hop reasoning and vector similarity search.

---

## 📐 System Architecture & Data Flow

The system is decoupled into three primary execution layers: the **Ingestion & Extraction Pipeline**, the **Hybrid Storage Engine**, and the **Synthesis & Serving API**.

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
