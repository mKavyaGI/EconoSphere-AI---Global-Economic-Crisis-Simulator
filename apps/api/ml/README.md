# EconoSphere AI - Machine Learning Data Pipeline Foundation

This directory houses the foundational machine learning dataset repositories, automation scripts, and exploratory notebooks for **EconoSphere AI** (Phase 11). It operates alongside the reusable ML execution framework located in `apps/api/app/ml/` and specialized forecasting architectures in `apps/api/app/forecasting/`.

---

## Directory Structure & Architecture

```
ml/
├── datasets/           # Managed data layers for analytical and predictive pipelines
│   ├── raw/            # Unmodified, immutable source data ingestion dumps
│   ├── interim/        # Partially cleaned or standardized temporary artifacts
│   └── processed/      # Fully validated, engineered features ready for model training
├── scripts/            # Executable processing, retrieval, and validation utilities
│   ├── download/       # Scripts for retrieving external economic indicator feeds
│   ├── preprocess/     # Data cleansing, transformation, and feature engineering routines
│   └── validation/     # Automated data quality checks, schema validation, and profiling
├── notebooks/          # Exploratory data analysis (EDA) and prototyping workspaces
└── README.md           # Documentation for the ML data pipeline hierarchy
```

---

## Folder Purposes & Workflow Guidelines

### 1. `datasets/`
The dataset layer strictly segregates data by stage of transformation to guarantee reproducibility, traceability, and auditability across all predictive experiments.
* **`raw/`**: Contains original, unmodified dataset artifacts (such as CSV, Parquet, or JSON dumps retrieved from macroeconomic databases, World Bank, or IMF APIs). **Rule:** Files in `raw/` must treat historical entries as read-only and immutable; never edit raw files manually or programmatically.
* **`interim/`**: Contains intermediate datasets currently undergoing multi-step transformation pipelines, exploratory cleaning, or schema alignment. Useful for debugging complex transformations without reprocessing raw inputs from scratch.
* **`processed/`**: Contains clean, fully normalized, and engineered feature tables (preferentially stored as high-performance Parquet files) that directly serve as inputs to `DatasetLoader` instances during model training and evaluation.

### 2. `scripts/`
Houses modular, standalone Python scripts engineered for reproducible execution, scheduled cron tasks, or continuous integration pipelines.
* **`download/`**: Automation routines dedicated to securely pulling macroeconomic dataset feeds from external APIs and depositing immutable snapshots directly into `datasets/raw/`.
* **`preprocess/`**: Transformation scripts that read from `raw/` or `interim/`, execute feature engineering using our reusable ML infrastructure (`app.ml.data.preprocessing`), and export finalized target matrices to `datasets/processed/`.
* **`validation/`**: Quality assurance utilities enforcing rigorous data contracts, temporal sequence continuity, null-value limits, and numerical distribution checks before models are optimized.

### 3. `notebooks/`
Jupyter notebooks specifically tailored for exploratory data analysis (EDA), correlation profiling, and rapid interactive prototyping of preprocessing transformation logic before consolidating code into production scripts or backend packages.

---

## Integration with FastAPI & Phase 11 Architecture

This pipeline layout cleanly decouples heavy dataset operational storage from the active application backend server:
- **Zero Runtime Interference**: The FastAPI application source code (`apps/api/app/`) remains completely independent of heavy dataset dumps, guaranteeing lean API server boot times and container builds.
- **Seamless Framework Integration**: Scripts structured within `scripts/` directly import robust type-checked classes from `app.ml` (such as `DatasetLoader`, `StandardScaler`, and `MissingValueImputer`) to prevent code duplication and enforce SOLID design principles.
- **Version Control Discipline**: Empty structure directories are preserved in Git via `.gitkeep` placeholder files, while voluminous raw, interim, and processed data artifacts are excluded from version control tracking via `.gitignore` policies.
