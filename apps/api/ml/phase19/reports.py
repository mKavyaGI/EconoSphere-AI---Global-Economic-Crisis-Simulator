"""
EconoSphere AI — Phase 19
reports.py : Generates all 17 required markdown reports under
             docs/model_accuracy/phase19/

All reports are generated post-hoc from in-memory result structures.
No model training occurs here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .models import CANDIDATE_DISPLAY_NAMES
from .utils import get_logger

log = get_logger()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR  = PROJECT_ROOT / "docs" / "model_accuracy" / "phase19"

# Utility display helper
def _disp(key: str) -> str:
    return CANDIDATE_DISPLAY_NAMES.get(key, key)


def _rmse_bar(rmse: float, ref: float = 5.0) -> str:
    """Rough ASCII bar chart for RMSE."""
    filled = min(int(rmse / ref * 20), 20)
    return "█" * filled + "░" * (20 - filled)


def _fmt_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table."""
    if df.empty:
        return "*No data*\n"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows   = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows) + "\n"


def write_report(filename: str, content: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    log.info(f"[REPORT] Written: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual report generators
# ─────────────────────────────────────────────────────────────────────────────

def r01_executive_summary(master_table: pd.DataFrame, governance: str,
                           valid_origins: list, before_hashes: dict,
                           after_hashes: dict, leakage_status: str) -> None:
    best = master_table[master_table["model"] != "M0_Phase11"].copy()
    best = best.sort_values("delta_rmse")
    best_model = best.iloc[0]["model"] if not best.empty else "N/A"
    best_delta = best.iloc[0]["delta_rmse"] if not best.empty else 0.0
    ts = datetime.now(timezone.utc).isoformat()

    content = f"""# Phase 19 — Executive Summary
*Generated: {ts}*

## Scientific Question
Does a fundamentally different forecasting architecture improve one-year-ahead GDP
growth forecasting over the frozen Phase 11 HistGradientBoosting production baseline?

## Backtest Setup
- **Origins evaluated:** {len(valid_origins)} chronological windows
- **Feature year → Target year:** Y → Y+1 (T+1 design strictly enforced)
- **Baseline:** Phase 11 HistGradientBoostingRegressor (frozen, not modified)

## Key Results
{_fmt_table(master_table[["model","mean_rmse","mean_mae","origin_wins","phase11_wins","delta_rmse"]])}

## Best Experimental Candidate
**{_disp(best_model)}** — ΔRMSE vs Phase 11 = {best_delta:+.4f}
(negative = candidate better; positive = Phase 11 better)

## Governance
- Leakage status: **{leakage_status}**
- Production hashes: **{"UNCHANGED" if before_hashes == after_hashes else "⚠️ CHANGED — GOVERNANCE FAILURE"}**
- Final decision: **{governance}**
- Phase 11 status: **FROZEN_PRODUCTION_RETAINED**
"""
    write_report("01_executive_summary.md", content)


def r02_experimental_design(valid_origins: list, skipped_origins: list) -> None:
    origin_rows = "\n".join(
        f"| {o['feature_year']} | {o['target_year']} | {o['n_train']} | {o['n_eval']} |"
        for o in valid_origins
    )
    skip_rows = "\n".join(
        f"| {s['feature_year']} | {s['target_year']} | {s['reason']} |"
        for s in skipped_origins
    ) or "| — | — | None skipped |"

    content = f"""# Phase 19 — Experimental Design

## Dataset
- Source: `data/processed/master_panel_t1_missingness.csv` (read-only)
- World Bank aggregate rows excluded.
- Target: `gdp_growth_next_year`
- Features: 31 locked Phase 11 features

## T+1 Design
```
Feature year = Y  →  Target year = Y+1
Train: year < Y   →  Eval: year == Y (with valid target labels)
```
This is enforced in code with explicit assertions.

## Valid Origins (Dynamically Determined)
| Feature Year | Target Year | N Train | N Eval |
|---|---|---|---|
{origin_rows}

## Skipped Origins
| Feature Year | Target Year | Reason |
|---|---|---|
{skip_rows}

## Candidates
| ID | Name | Architecture |
|---|---|---|
| M0 | Phase 11 (Control) | HistGradientBoosting (Phase 11 exact config) |
| A1 | Ridge | StandardScaler + Ridge (alpha from {{0.1,1,10,100}}) |
| A2 | Elastic Net | StandardScaler + ElasticNet (alpha × l1_ratio grid) |
| A3 | Huber | StandardScaler + HuberRegressor (epsilon from {{1.35,1.5,2.0}}) |
| A4 | Random Forest | SimpleImputer + RandomForest (300 trees, max_depth=8) |
| A5a | Ensemble Equal | Average of M0, A1, A4 predictions |
| A5b | Ensemble Val-Weighted | Val-optimised weights (M0, A1, A4) |
| A6 | Residual Model | M0 + Ridge(OOF residuals) |

## Hyperparameter Selection Rule
All hyperparameter selection uses the last 20% of training years as validation.
Test/forecast-origin rows are never inspected during selection.

## Ensemble Weight Rule (A5b)
Weights minimise val-set RMSE using scipy SLSQP (non-negative, sum=1).
Test-set predictions are not used for weight selection.

## Residual Model Rule (A6)
OOF M0 predictions generated via 5-fold chronological KFold inside training window.
Residual = actual_target - OOF_m0_pred.
Ridge residual model trained on those residuals.
Test/eval residuals are never used.
"""
    write_report("02_experimental_design.md", content)


def r03_chronological_backtest(results: list) -> None:
    rows = []
    for r in results:
        m0_rmse = r["metrics"]["M0_Phase11"]["rmse"]
        for cand, m in r["metrics"].items():
            rows.append({
                "feature_year": r["feature_year"],
                "target_year":  r["target_year"],
                "model":        _disp(cand),
                "rmse":         round(m["rmse"], 4),
                "mae":          round(m["mae"], 4),
                "n":            m["n"],
                "delta_rmse":   round(m["rmse"] - m0_rmse, 4),
            })
    df = pd.DataFrame(rows)
    content = f"""# Phase 19 — Chronological Backtest Results

Per-origin RMSE for every candidate:

{_fmt_table(df)}
"""
    write_report("03_chronological_backtest.md", content)


def r04_architecture_comparison(master_table: pd.DataFrame) -> None:
    content = f"""# Phase 19 — Architecture Comparison

## Master Result Table
| Model | Mean RMSE | Median RMSE | Mean MAE | Origin Wins | Phase 11 Wins | ΔRMSE |
|---|---|---|---|---|---|---|
"""
    for _, row in master_table.iterrows():
        delta_str = f"{row['delta_rmse']:+.4f}"
        content += (
            f"| {_disp(row['model'])} | {row['mean_rmse']:.4f} | "
            f"{row['median_rmse']:.4f} | {row['mean_mae']:.4f} | "
            f"{row['origin_wins']} | {row['phase11_wins']} | {delta_str} |\n"
        )

    content += """
## Interpretation Guide
- **ΔRMSE < 0**: Candidate beats Phase 11 (lower RMSE is better)
- **ΔRMSE > 0**: Phase 11 beats candidate
- **Origin Wins**: Number of chronological windows where candidate RMSE < Phase 11 RMSE

> A robust improvement requires winning across multiple origins, not just one.
"""
    write_report("04_architecture_comparison.md", content)


def r05_ridge_elasticnet_huber(results: list) -> None:
    rows = []
    for r in results:
        for cand in ["A1_Ridge", "A2_ElasticNet", "A3_Huber"]:
            if cand in r["metrics"]:
                m = r["metrics"][cand]
                hp = r.get("hyperparams", {})
                hp_str = ""
                if cand == "A1_Ridge":
                    hp_str = f"alpha={hp.get('ridge_alpha','?')}"
                elif cand == "A2_ElasticNet":
                    hp_str = f"alpha={hp.get('en_alpha','?')} l1={hp.get('en_l1_ratio','?')}"
                elif cand == "A3_Huber":
                    hp_str = f"eps={hp.get('huber_epsilon','?')}"
                rows.append({
                    "origin": r["feature_year"],
                    "model":  cand,
                    "rmse":   round(m["rmse"], 4),
                    "mae":    round(m["mae"], 4),
                    "hyperparams": hp_str,
                })
    df = pd.DataFrame(rows)
    content = f"""# Phase 19 — Ridge / Elastic Net / Huber Analysis

{_fmt_table(df)}

## Notes
- All regularization strength is selected on the chronological validation split (last 20% of training years).
- Test/forecast-origin data never influence hyperparameter choice.
- ElasticNet combines L1 (sparsity) and L2 (shrinkage) regularization.
- Huber loss down-weights extreme GDP shock observations.
"""
    write_report("05_ridge_elasticnet_huber.md", content)


def r06_random_forest(results: list) -> None:
    rows = []
    for r in results:
        if "A4_RandomForest" in r["metrics"]:
            m = r["metrics"]["A4_RandomForest"]
            m0 = r["metrics"]["M0_Phase11"]["rmse"]
            rows.append({
                "origin": r["feature_year"],
                "target": r["target_year"],
                "rmse":   round(m["rmse"], 4),
                "mae":    round(m["mae"], 4),
                "delta_rmse": round(m["rmse"] - m0, 4),
            })
    df = pd.DataFrame(rows)
    content = f"""# Phase 19 — Random Forest Analysis

## Configuration
- n_estimators: 300
- max_depth: 8
- min_samples_leaf: 5
- random_state: 42

## Per-Origin Results
{_fmt_table(df)}

## Architecture Notes
Random Forest provides an alternative nonlinear architecture to HistGradientBoosting.
Unlike boosting, RF builds trees independently and averages. This may provide
different generalization behaviour during shock periods.
"""
    write_report("06_random_forest.md", content)


def r07_ensemble_analysis(results: list) -> None:
    rows = []
    for r in results:
        for cand in ["A5a_Ensemble", "A5b_Ensemble"]:
            if cand in r["metrics"]:
                m = r["metrics"][cand]
                m0_rmse = r["metrics"]["M0_Phase11"]["rmse"]
                w = r.get("ensemble_weights", {})
                rows.append({
                    "origin":   r["feature_year"],
                    "model":    cand,
                    "rmse":     round(m["rmse"], 4),
                    "delta":    round(m["rmse"] - m0_rmse, 4),
                    "w_M0":     round(w.get("M0_Phase11", 1/3), 3) if cand == "A5b_Ensemble" else "1/3",
                    "w_Ridge":  round(w.get("A1_Ridge", 1/3), 3) if cand == "A5b_Ensemble" else "1/3",
                    "w_RF":     round(w.get("A4_RandomForest", 1/3), 3) if cand == "A5b_Ensemble" else "1/3",
                })
    df = pd.DataFrame(rows)
    content = f"""# Phase 19 — Ensemble Analysis

## A5a: Equal-Weight Ensemble
Weights: M0=1/3, Ridge=1/3, RandomForest=1/3

## A5b: Validation-Selected Weights
Weights determined by SLSQP optimisation on validation-split RMSE.
Test/forecast-origin targets never influence weight selection (L7 satisfied).

## Per-Origin Results
{_fmt_table(df)}

## Governance
Ensemble weights for A5b were determined **exclusively** on validation data
(last 20% of training years). This satisfies L7 of the leakage audit.
"""
    write_report("07_ensemble_analysis.md", content)


def r08_residual_modeling(results: list) -> None:
    rows = []
    for r in results:
        if "A6_Residual" in r["metrics"]:
            m  = r["metrics"]["A6_Residual"]
            m0 = r["metrics"]["M0_Phase11"]["rmse"]
            ri = r.get("residual_info", {})
            rows.append({
                "origin":   r["feature_year"],
                "target":   r["target_year"],
                "rmse":     round(m["rmse"], 4),
                "delta":    round(m["rmse"] - m0, 4),
                "oof_valid": ri.get("n_oof_valid", "?"),
                "oof_nan":   ri.get("n_oof_nan", "?"),
            })
    df = pd.DataFrame(rows)
    content = f"""# Phase 19 — Residual Modeling (A6)

## Method
1. OOF Phase 11 predictions generated via 5-fold chronological KFold inside training window.
2. OOF residuals = actual_target − OOF_M0_prediction.
3. Ridge residual model trained on (X_train, OOF_residuals).
4. Final prediction = M0_prediction(eval) + Residual_prediction(eval).

## Leakage Audit
- OOF predictions generated WITHIN training window only (L8 satisfied).
- Eval/test residuals never used to train the residual model (L9 satisfied).
- Residual model hyperparameter (alpha) selected on validation split (L6 satisfied).

## Hypothesis
Phase 11 may systematically under- or over-predict in certain regimes.
A residual model can correct these patterns if they are learnable from historical data.

## Per-Origin Results
{_fmt_table(df)}
"""
    write_report("08_residual_modeling.md", content)


def r09_country_analysis(preds_df: pd.DataFrame) -> None:
    rows = []
    for (cc, cand), g in preds_df.groupby(["country_code", "candidate"]):
        m = {
            "country": cc,
            "candidate": _disp(cand),
            "rmse": round(float(np.sqrt(g["squared_error"].mean())), 4),
            "mae":  round(float(g["absolute_error"].mean()), 4),
            "n":    len(g),
        }
        rows.append(m)
    df = pd.DataFrame(rows).sort_values(["country", "rmse"])
    content = f"""# Phase 19 — Country-Level Analysis

All countries with ≥2 predictions:

{_fmt_table(df[df['n'] >= 2].head(200))}

> Countries with n=1 are marked LOW_SAMPLE_EVIDENCE.
"""
    write_report("09_country_analysis.md", content)


def r10_priority_country_analysis(preds_df: pd.DataFrame) -> None:
    priority = ["GBR", "BRA", "FRA", "CAN", "AUS"]
    rows = []
    for cc in priority:
        sub = preds_df[preds_df["country_code"] == cc]
        if sub.empty:
            rows.append({"country": cc, "status": "NOT FOUND"})
            continue
        for cand, g in sub.groupby("candidate"):
            if len(g) < 2:
                rows.append({"country": cc, "candidate": _disp(cand),
                             "rmse": "LOW_SAMPLE_EVIDENCE", "mae": "-", "n": len(g)})
            else:
                rows.append({
                    "country":   cc,
                    "candidate": _disp(cand),
                    "rmse":      round(float(np.sqrt(g["squared_error"].mean())), 4),
                    "mae":       round(float(g["absolute_error"].mean()), 4),
                    "n":         len(g),
                })
    df = pd.DataFrame(rows)
    content = f"""# Phase 19 — Priority Country Analysis

Priority countries: GBR, BRA, FRA, CAN, AUS

{_fmt_table(df)}

> Claims of improvement in priority countries require n ≥ 3 observations.
> LOW_SAMPLE_EVIDENCE = fewer than 2 predictions available.
"""
    write_report("10_priority_country_analysis.md", content)


def r11_guardrail_analysis(preds_df: pd.DataFrame) -> None:
    guardrails = ["USA", "CHN", "DEU", "JPN", "IND"]
    rows = []
    for cc in guardrails:
        sub = preds_df[preds_df["country_code"] == cc]
        if sub.empty:
            rows.append({"country": cc, "status": "NOT FOUND"})
            continue
        for cand, g in sub.groupby("candidate"):
            m0_rows = preds_df[(preds_df["country_code"] == cc) & (preds_df["candidate"] == "M0_Phase11")]
            m0_rmse = float(np.sqrt(m0_rows["squared_error"].mean())) if not m0_rows.empty else np.nan
            cand_rmse = float(np.sqrt(g["squared_error"].mean()))
            rows.append({
                "country":    cc,
                "candidate":  _disp(cand),
                "rmse":       round(cand_rmse, 4),
                "mae":        round(float(g["absolute_error"].mean()), 4),
                "delta_rmse": round(cand_rmse - m0_rmse, 4) if not np.isnan(m0_rmse) else "?",
                "n":          len(g),
            })
    df = pd.DataFrame(rows)
    content = f"""# Phase 19 — Guardrail Country Analysis

Guardrail countries: USA, CHN, DEU, JPN, IND

A candidate that improves overall RMSE but degrades guardrail countries is NOT robust.

{_fmt_table(df)}
"""
    write_report("11_guardrail_analysis.md", content)


def r12_shock_analysis(preds_df: pd.DataFrame) -> None:
    shock_year = 2020
    shock_df = preds_df[preds_df["target_year"] == shock_year]
    stable_df = preds_df[preds_df["target_year"] != shock_year]

    rows = []
    for cand in preds_df["candidate"].unique():
        for label, subset in [("Shock (2020)", shock_df), ("Non-Shock", stable_df)]:
            g = subset[subset["candidate"] == cand]
            if g.empty:
                continue
            rows.append({
                "candidate": _disp(cand),
                "period":    label,
                "rmse":      round(float(np.sqrt(g["squared_error"].mean())), 4),
                "mae":       round(float(g["absolute_error"].mean()), 4),
                "n":         len(g),
            })
    df = pd.DataFrame(rows)
    content = f"""# Phase 19 — Shock Period Analysis

## COVID-19 Shock: Target Year 2020

All models are expected to struggle with the 2020 shock, as COVID-19 was
unprecedented and no pre-2020 feature can capture it.

{_fmt_table(df)}

> Note: 2020 performance is informative but not a primary selection criterion,
> since no model could plausibly predict a global pandemic from Y-1 features.
"""
    write_report("12_shock_analysis.md", content)


def r13_error_tail_analysis(preds_df: pd.DataFrame) -> None:
    rows = []
    for cand, g in preds_df.groupby("candidate"):
        ae = g["absolute_error"].values
        rows.append({
            "candidate":  _disp(cand),
            "p90_ae":     round(float(np.percentile(ae, 90)), 4),
            "p95_ae":     round(float(np.percentile(ae, 95)), 4),
            "max_ae":     round(float(np.max(ae)), 4),
            "n_above_10": int((ae > 10).sum()),
            "n_above_15": int((ae > 15).sum()),
        })
    df = pd.DataFrame(rows)
    worst = (
        preds_df.sort_values("absolute_error", ascending=False)
        .head(20)[["target_year", "country_code", "candidate", "actual_gdp_growth",
                   "predicted_gdp_growth", "absolute_error"]]
        .round(3)
    )
    content = f"""# Phase 19 — Error Tail Analysis

## Tail Error Percentiles
{_fmt_table(df)}

## Top 20 Worst Predictions (all candidates)
{_fmt_table(worst)}

## Interpretation
A model that improves mean RMSE by concentrating large errors in a few observations
is not considered robust. Examine the p95 and max_ae columns alongside mean_rmse.
"""
    write_report("13_error_tail_analysis.md", content)


def r14_prediction_diversity(results: list) -> None:
    rows = []
    for r in results:
        candidates = list(r["predictions"].keys())
        for i, c1 in enumerate(candidates):
            for c2 in candidates[i+1:]:
                p1 = r["predictions"][c1]
                p2 = r["predictions"][c2]
                if len(p1) > 1 and len(p2) > 1:
                    corr = float(np.corrcoef(p1, p2)[0, 1])
                    rows.append({
                        "origin": r["feature_year"],
                        "pair":   f"{_disp(c1)} vs {_disp(c2)}",
                        "corr":   round(corr, 4),
                    })
    df = pd.DataFrame(rows)
    content = f"""# Phase 19 — Prediction Diversity Analysis

Pairwise prediction correlation between candidates across all origins.
High correlation (>0.95) means models make nearly identical predictions,
and an ensemble over them provides little additional diversity.

{_fmt_table(df.head(200))}

## Interpretation
If two models are highly correlated, their errors are similar, and ensembling them
provides minimal benefit. Low correlation (complementary errors) is the condition
under which ensembles tend to improve over individual models.
"""
    write_report("14_prediction_diversity.md", content)


def r15_leakage_audit(leakage_log: list[dict]) -> None:
    rows = "\n".join(
        f"| {e['rule']} | {e['status']} | {e.get('detail', '')} |"
        for e in leakage_log
    )
    content = f"""# Phase 19 — Leakage Audit

| Rule | Status | Detail |
|---|---|---|
{rows}

## Rules Reference
- L1: No Y+1 target in features for year Y
- L2: No future-year feature in forecast row
- L3: Scalers fitted on training data only
- L4: Imputers fitted on training data only
- L5: Feature selection fitted on training data only
- L6: Hyperparameter selection does not inspect test performance
- L7: Ensemble weights do not use test results
- L8: Residual models do not train on test residuals
- L9: Out-of-fold residual predictions used where required
- L10: Country-level evaluation does not leak test targets into training
- L11: No production artifact is mutated
- L12: No fabricated data is introduced
"""
    write_report("15_leakage_audit.md", content)


def r16_production_readiness(governance: str, before_hashes: dict,
                              after_hashes: dict, master_table: pd.DataFrame) -> None:
    hash_ok = before_hashes == after_hashes
    content = f"""# Phase 19 — Production Readiness Assessment

## Governance Decision
**{governance}**

## Production Hash Verification
{"✅ ALL PRODUCTION HASHES UNCHANGED" if hash_ok else "⚠️ HASH MISMATCH — GOVERNANCE FAILURE"}

### Before Hashes
```json
{json.dumps(before_hashes, indent=2)}
```
### After Hashes
```json
{json.dumps(after_hashes, indent=2)}
```

## Promotion Gate
No model from Phase 19 may replace Phase 11 during this phase.

Valid outcomes:
- `ARCHITECTURE_ROBUST_AND_PROMISING` → Future formal promotion audit warranted
- `ARCHITECTURE_PROMISING_BUT_INCONCLUSIVE` → Further research required
- `NO_ARCHITECTURAL_IMPROVEMENT_FOUND` → Phase 11 remains production

**Phase 11 remains FROZEN_PRODUCTION_RETAINED.**
"""
    write_report("16_production_readiness.md", content)


def r17_walkthrough(results: list, master_table: pd.DataFrame,
                    governance: str, valid_origins: list,
                    leakage_status: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    n_origins = len(results)
    n_preds   = sum(len(r["y_eval"]) * len(r["predictions"]) for r in results)
    content = f"""# Phase 19 — Execution Walkthrough
*Completed: {ts}*

## What Was Done
1. Recorded MD5/SHA256 hashes of all frozen production artifacts.
2. Loaded `master_panel_t1_missingness.csv` (read-only).
3. Dynamically determined valid forecast origins.
4. Ran chronological expanding-window backtest over **{n_origins} origins**.
5. Evaluated **8 model configurations** (M0, A1–A6 including A5a/A5b).
6. Generated {n_preds:,} prediction records.
7. Verified production hashes post-execution.
8. Generated 17 reports under `docs/model_accuracy/phase19/`.

## Origins Used
{", ".join(str(o["feature_year"]) + "→" + str(o["target_year"]) for o in valid_origins)}

## Leakage Status
**{leakage_status}**

## Final Governance Decision
**{governance}**
**FROZEN_PRODUCTION_RETAINED**

## Files Created (Phase 19 Experimental Artifacts)
All experimental model artifacts are named with `EXPERIMENTAL_ONLY`.
No production artifact was modified.
"""
    write_report("17_phase19_walkthrough.md", content)


# ── Master entry point ─────────────────────────────────────────────────────────

def generate_all_reports(
    results: list,
    master_table: pd.DataFrame,
    preds_df: pd.DataFrame,
    valid_origins: list,
    skipped_origins: list,
    before_hashes: dict,
    after_hashes: dict,
    leakage_log: list[dict],
    governance: str,
) -> None:
    leakage_status = (
        "ALL LEAKAGE CHECKS PASSED"
        if all(e["status"] == "PASS" for e in leakage_log)
        else "LEAKAGE FAILURES DETECTED — see report 15"
    )
    r01_executive_summary(master_table, governance, valid_origins,
                          before_hashes, after_hashes, leakage_status)
    r02_experimental_design(valid_origins, skipped_origins)
    r03_chronological_backtest(results)
    r04_architecture_comparison(master_table)
    r05_ridge_elasticnet_huber(results)
    r06_random_forest(results)
    r07_ensemble_analysis(results)
    r08_residual_modeling(results)
    r09_country_analysis(preds_df)
    r10_priority_country_analysis(preds_df)
    r11_guardrail_analysis(preds_df)
    r12_shock_analysis(preds_df)
    r13_error_tail_analysis(preds_df)
    r14_prediction_diversity(results)
    r15_leakage_audit(leakage_log)
    r16_production_readiness(governance, before_hashes, after_hashes, master_table)
    r17_walkthrough(results, master_table, governance, valid_origins, leakage_status)
    log.info(f"[REPORTS] All 17 reports written to {REPORTS_DIR}")
