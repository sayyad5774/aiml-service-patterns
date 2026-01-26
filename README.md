# Acosta AIML Test Repo

Three mini-products:
- apps/churn: churn model train -> MLflow -> FastAPI endpoint
- apps/rag_policy: policy Q&A with RAG + tool calling stub -> FastAPI
- apps/anomaly_ops: ops anomaly detection + runbook stub -> FastAPI

## Quickstart (local)
1) Create venv and install each app:
   - cd apps/churn && pip install -e .
   - cd ../rag_policy && pip install -e .
   - cd ../anomaly_ops && pip install -e .

2) Run:
   - Churn: python -m churn.train && python -m churn.serve
   - RAG:   python -m rag_policy.ingest && python -m rag_policy.serve
   - Anom:  python -m anomaly_ops.detect && python -m anomaly_ops.serve


> **Note:** This repository was bootstrapped as a multi-service skeleton and
> iteratively refined. Some directories share early commit history by design.
> Subsequent changes are scoped and intentional.


# AIML Service Patterns

Reference implementations for applied AI/ML services, focusing on
training pipelines, inference APIs, testing, and CI.

This repository is intentionally structured as a **patterns library**
rather than a single application. Each service demonstrates a common
approach used in production-adjacent AIML systems.

---

## What This Repository Demonstrates

- Model training and artifact management
- Typed inference services using FastAPI
- Integration testing for ML-backed APIs
- Reproducible local development workflows
- CI validation via GitHub Actions

The code is designed to be **clear, inspectable, and extensible** rather
than production-hardened.

---

## Repository Structure

apps/
churn/ # Churn prediction training + inference service
docs/ # Design notes and future expansions
infra/terraform/ # Infrastructure scaffolding (local-focused)
scripts/ # Utility and dev scripts
shared/ # Shared patterns and helpers


---

## Churn Prediction Service

The churn service demonstrates an end-to-end ML workflow:

- Synthetic data generation
- Model training with scikit-learn
- Artifact persistence
- FastAPI inference endpoint
- API integration tests with pytest

### Local Quickstart

```bash
cd apps/churn

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

python -m churn.train
python -m churn.serve
