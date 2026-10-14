# 🌐 EconoSphere AI
### *Global Economic Crisis Simulator & Decision Intelligence Platform*

[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2014-black?logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791?logo=postgresql)](https://www.postgresql.org/)
[![Neo4j](https://img.shields.io/badge/Graph_DB-Neo4j-008CC1?logo=neo4j)](https://neo4j.com/)
[![Redis](https://img.shields.io/badge/Cache-Redis-DC382D?logo=redis)](https://redis.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)

---

## 📌 Overview

**EconoSphere AI** is a comprehensive Digital Twin of the Global Economy. It empowers policymakers, analysts, and researchers to:
- **Simulate Crisis Scenarios**: Model supply-chain shocks, trade tariffs, interest rate spikes, and pandemic disruptions in real time.
- **Forecast Macroeconomic Trends**: Predict multi-horizon GDP growth, inflation, unemployment, and trade balances with high-precision machine learning models.
- **Analyze Global Trade Networks**: Explore interactive graph visualizations of international trade dependencies, bilateral flows, and vulnerability contagion pathways.
- **Generate AI Policy Interventions**: Receive automated, data-driven monetary and fiscal policy recommendations tailored to mitigated systemic economic risks.

---

## 🏗️ Repository Architecture

This repository is organized as a monorepo:

```text
EconoSphere-AI/
├── apps/
│   ├── web/                    # Next.js 14 frontend (React 19, TypeScript, ECharts, Cytoscape, Leaflet)
│   └── api/                    # FastAPI backend gateway & microservices
├── packages/
│   ├── shared/                 # Shared data schemas, utilities, and exceptions
│   └── ui/                     # Reusable design system UI components
├── infrastructure/
│   └── docker/                 # Docker Compose orchestrating Postgres, Neo4j, Redis, API, and Web
├── data/
│   ├── raw/                    # Raw historical macroeconomic and bilateral trade datasets
│   ├── processed/              # Engineered features and panel datasets
│   ├── external/               # External reference data (ISO codes, WB/IMF series)
│   └── features/               # Normalized model feature vectors
├── models/                     # Trained ML model weights and benchmark artifacts
└── docs/                       # Architecture diagrams, API specs, and model accuracy audit reports
```

---

## 🚀 Quick Start Guide

### Prerequisites
Make sure you have the following installed on your machine:
- **Git**
- **Node.js** (v18.0 or newer) & **npm** / **pnpm**
- **Python** (v3.11 or newer) & **uv** or **pip**
- **Docker & Docker Compose** (recommended for running databases)

---

### Option 1: Instant Start with Docker (Recommended)

Spin up the entire stack (PostgreSQL, Neo4j, Redis, FastAPI backend, and Next.js frontend) with a single command:

```bash
# 1. Clone the repository
git clone https://github.com/mKavyaGI/EconoSphere-AI---Global-Economic-Crisis-Simulator.git
cd EconoSphere-AI---Global-Economic-Crisis-Simulator

# 2. Launch all services via Docker Compose
docker compose -f infrastructure/docker/docker-compose.yml up --build
```

- **Frontend Application**: `http://localhost:3000`
- **Backend API Docs (Swagger)**: `http://localhost:8000/docs`
- **Neo4j Browser**: `http://localhost:7474` (User: `neo4j`, Password: `password`)
- **Database Adminer**: `http://localhost:8080`

---

### Option 2: Local Development Setup

If you prefer to run services locally:

#### 1. Start the Supporting Databases
Run the required backing services (PostgreSQL, Neo4j, Redis):
```bash
docker compose -f infrastructure/docker/docker-compose.yml up -d postgres neo4j redis
```

#### 2. Backend Setup (FastAPI)
```bash
cd apps/api

# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies (using uv or pip)
pip install -r <(python -m pip freeze)  # or install via uv / pip install .
pip install -e .

# Configure environment variables
cp .env.example .env

# Run database migrations
alembic upgrade head

# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Backend API will be live at `http://localhost:8000`. Test docs at `http://localhost:8000/docs`.

#### 3. Frontend Setup (Next.js)
In a new terminal window:
```bash
cd apps/web

# Configure environment variables
cp .env.example .env.local

# Install dependencies
npm install

# Start Next.js development server
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## ⚙️ Environment Variables

### Backend (`apps/api/.env`)
| Variable | Description | Default |
|---|---|---|
| `POSTGRES_HOST` | PostgreSQL host | `127.0.0.1` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | Database name | `econosphere` |
| `POSTGRES_USER` | Database user | `postgres` |
| `POSTGRES_PASSWORD` | Database password | `postgres` |
| `NEO4J_URI` | Neo4j Bolt connection URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j user | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |
| `REDIS_HOST` | Redis cache host | `localhost` |
| `REDIS_PORT` | Redis cache port | `6379` |
| `JWT_SECRET` | Secret key for JWT signing | `change-in-production` |

### Frontend (`apps/web/.env.local`)
| Variable | Description | Default |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Base endpoint of FastAPI backend | `http://localhost:8000/api/v1` |

---

## 🤝 Contribution Workflow for Teammates

1. **Pull the latest changes from `main`**:
   ```bash
   git checkout main
   git pull origin main
   ```
2. **Create a new feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```
3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat(module): add meaningful description of changes"
   ```
4. **Push your branch and open a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

---

## 📜 License & Acknowledgments
Developed with ❤️ by the **EconoSphere AI Team**.
Specialized simulation models trained on World Bank, IMF, and UN Comtrade economic data.
