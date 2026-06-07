<div align="center">

# Self Evolving Research Agent

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Database-Neo4j_Aura-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![FAISS](https://img.shields.io/badge/Vector_Index-FAISS-7A8494?style=flat-square&logo=cpu&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.0_Flash-228B22?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![ONNX](https://img.shields.io/badge/Inference-ONNX_Runtime-005CBB?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)

<img width="812" height="541" alt="Self Evolving Research Agent" src="https://github.com/user-attachments/assets/ce20ad31-7171-4716-ba50-3a07ab22df79" />

*A knowledge graph that learns while you sleep.*

</div>

**[Home](#)** | [Usage](USAGE.md) | [Contributing]()

---

Most RAG systems are frozen in time — you feed them documents once and they stay dumb. This one doesn't. Every few hours it reaches out to arXiv and RSS feeds, pulls the latest papers on topics you care about, extracts structured knowledge triples, and weaves them into a live Neo4j graph. Ask it a question tomorrow and it knows things it didn't know today.

---

<div align="center">
<sub>Built with Gemini · Neo4j · FAISS · FastAPI · Prefect</sub>
</div>

