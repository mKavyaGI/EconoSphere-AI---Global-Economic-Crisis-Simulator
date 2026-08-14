# Software Architecture

The EconoSphere AI platform is built as a **Modular Monolith** using the following stack:
* **Frontend:** Next.js, Tailwind CSS, shadcn/ui
* **Backend Gateway:** FastAPI (Python)
* **Databases:** PostgreSQL (Structured), Neo4j (Graph/Trade Networks), Redis (Cache)

## Module Layout
- `services/simulation`: Engine to run macro events.
- `services/forecasting`: AI time-series models.
- `services/graph`: Trade relationship analysis.
- `services/policy`: Recommendation engine.
