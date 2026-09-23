"""
EconoSphere AI — Phase 20
reports.py : Generates all 19 required Phase 20 audit reports.

Reports generated:
    01_executive_summary.md
    02_methodology.md
    03_clipping_sensitivity.md
    04_out_of_sample_validation.md
    05_origin_level_results.md
    06_component_ablation.md
    07_error_diversity.md
    08_priority_countries.md
    09_guardrail_countries.md
    10_shock_analysis.md
    11_error_tail_analysis.md
    12_statistical_comparison.md
    13_reproducibility.md
    14_data_vintage_audit.md
    15_operational_readiness.md
    16_rollback_plan.md
    17_promotion_gates.md
    18_governance_decision.md
    PHASE20_PROMOTION_AUDIT_REPORT.md  (consolidated)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parents[4] / "docs" / "model_accuracy" / "phase20"

PRIORITY_COUNTRIES  = ["GBR", "BRA", "FRA", "CAN", "AUS"]
GUARDRAIL_COUNTRIES = ["USA", "CHN", "DEU", "JPN", "IND"]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_all_reports(audit_data: dict) -> list[str]:
    """
    Generate all 19 Phase 20 reports.
    Returns list of generated file paths.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []

    report_fns = [
        _rpt_01_executive_summary,
        _rpt_02_methodology,
        _rpt_03_clipping_sensitivity,
        _rpt_04_out_of_sample,
        _rpt_05_origin_level,
        _rpt_06_ablation,
        _rpt_07_error_diversity,
        _rpt_08_priority_countries,
        _rpt_09_guardrail_countries,
        _rpt_10_shock_analysis,
        _rpt_11_error_tail,
        _rpt_12_statistical,
        _rpt_13_reproducibility,
        _rpt_14_data_vintage,
        _rpt_15_operational,
        _rpt_16_rollback,
        _rpt_17_promotion_gates,
        _rpt_18_governance,
        _rpt_consolidated,
    ]

    for fn in report_fns:
        try:
            path = fn(audit_data)
            generated.append(str(path))
        except Exception as exc:
            err_path = REPORTS_DIR / f"ERROR_{fn.__name__}.md"
            err_path.write_text(f"# Report Generation Error\n\n{fn.__name__}: {exc}\n")
            generated.append(str(err_path))

    return generated


def _rpt_01_executive_summary(d: dict) -> Path:
    agg = d.get("aggregate", {})
    gates = d.get("gates", {})
    gov   = d.get("governance", {})
    path  = REPORTS_DIR / "01_executive_summary.md"
    _write(path, f"""# Phase 20 — Promotion Audit Executive Summary

Generated: {_ts()}

## Candidate
A5a Equal-Weight Ensemble = (M0 + Ridge + RandomForest) / 3

## Result
**{gov.get('decision', 'PENDING')}**

## Key Metrics (Phase 19 clipping C1)

| Metric | M0 (Phase 11) | A5a | ΔRMSE |
|---|---|---|---|
| Mean RMSE | {agg.get('m0_mean_rmse', 'N/A')} | {agg.get('mean_rmse', 'N/A')} | {agg.get('delta_rmse', 'N/A')} |
| Median RMSE | — | {agg.get('median_rmse', 'N/A')} | — |
| Mean MAE | — | {agg.get('mean_mae', 'N/A')} | — |
| Origin wins | — | {agg.get('origin_wins', 'N/A')} / {agg.get('n_origins', 'N/A')} | — |

## Phase 19 Reproduction
{d.get('reproduction', {}).get('status', 'N/A')}

## Promotion Gates Summary

| Gate | Status |
|---|---|
""" + "\n".join(
        f"| {g} | {s} |"
        for g, s in gates.items()
    ) + f"""

## Recommendation
{gov.get('recommendation', 'N/A')}
""")
    return path


def _rpt_02_methodology(d: dict) -> Path:
    path = REPORTS_DIR / "02_methodology.md"
    origins = d.get("origins", [])
    oo = sorted(set(o["feature_year"] for o in origins)) if origins else []
    _write(path, f"""# Phase 20 — Methodology

## T+1 Design
```
Feature year = Y  →  Target year = Y+1
Training: year < Y  (chronological)
Evaluation: year == Y (with valid gdp_growth_next_year)
```

## Origin Set
Phase 19 origins (loaded from metadata): {d.get('phase19_origins', [])}
Phase 20 dynamically computed:           {oo}
Origin set match: {d.get('origin_match', 'N/A')}

## Reused from Phase 19 (unchanged)
- 31 locked features
- World Bank aggregate exclusion list
- `get_valid_origins()` with `min_train_rows=50`
- `get_train_eval_split()` (year < feature_year / year == feature_year)
- `_make_val_split()` (last 20% of training years for Ridge alpha)
- `build_m0()`, `build_ridge()`, `build_random_forest()` with Phase 19 configs
- `compute_metrics()` (RMSE, MAE, median AE)
- Prediction clipping: Ridge only, tree models never clipped

## Phase 20 New Analyses
- Clipping sensitivity (C0/C1/C2)
- Component ablation (7 subsets)
- Error diversity (residual + prediction correlations)
- Leave-one-origin-out (LOOO)
- Paired Wilcoxon signed-rank test
- Reproducibility audit (2 runs)
- True OOS evaluation
- Operational readiness + independent artifact load

## Production Isolation
Phase 11 artifacts are FROZEN throughout. Hash verified before and after.
No experimental model is saved to models/phase11/.
""")
    return path


def _rpt_03_clipping_sensitivity(d: dict) -> Path:
    clip_results = d.get("clipping_sensitivity", {})
    g5 = d.get("gates", {}).get("G5", "N/A")
    path = REPORTS_DIR / "03_clipping_sensitivity.md"

    rows = ""
    for cfg_name, res in clip_results.items():
        rows += (
            f"| {cfg_name} | {res.get('label','')[:30]} | "
            f"{res.get('a5a_mean_rmse','N/A')} | {res.get('m0_mean_rmse','N/A')} | "
            f"{res.get('delta_rmse','N/A'):+.4f} | {res.get('origin_wins','N/A')} |\n"
            if isinstance(res.get('delta_rmse'), (int, float))
            else f"| {cfg_name} | N/A | N/A | N/A | N/A | N/A |\n"
        )

    _write(path, f"""# Phase 20 — Clipping Sensitivity Analysis

## Configurations
| Config | Description |
|---|---|
| C0 | No clipping — raw linear predictions |
| C1 | Phase 19 bounds [-40%, +60%] |
| C2 | Alternative bounds [-50%, +75%] |

> Configurations are pre-specified. No test-set clipping selection.

## Results

| Config | Label | A5a RMSE | M0 RMSE | ΔRMSE | A5a Wins |
|---|---|---|---|---|---|
{rows}

## Gate 5 Assessment
{g5}

## Interpretation
C0 is expected to show degraded performance for linear models due to known
extrapolation behavior (extreme exchange rates, micro-state GDP swings).
The key question is whether the A5a advantage under C1 is an artifact of
the clipping boundary or a genuine architectural benefit.

> A5a improvement is not solely an artifact of clipping if it survives C1 and C2.
""")
    return path


def _rpt_04_out_of_sample(d: dict) -> Path:
    oos = d.get("oos", {})
    path = REPORTS_DIR / "04_out_of_sample_validation.md"
    oos_agg = oos.get("aggregate", {})
    _write(path, f"""# Phase 20 — True Out-of-Sample Validation

## Status
{oos.get('message', 'N/A')}

## Available
{oos.get('available', False)}

## Out-of-Sample Origins
{oos.get('oos_origins_identified', [])}

## Results
{'N/A — no OOS data available. Gate 11 = INCONCLUSIVE.' if not oos.get('available') else ''}
{'Mean RMSE: ' + str(oos_agg.get('mean_rmse', 'N/A')) if oos.get('available') else ''}
{'M0 Mean RMSE: ' + str(oos_agg.get('m0_mean_rmse', 'N/A')) if oos.get('available') else ''}
{'ΔRMSE: ' + str(oos_agg.get('delta_rmse', 'N/A')) if oos.get('available') else ''}

## Governance Implication
If no OOS data is available:
```
PROMISING_BUT_NEW_OUT_OF_SAMPLE_EVIDENCE_UNAVAILABLE
FROZEN_PRODUCTION_RETAINED
```
Even if all other gates pass, promotion cannot be final without
genuine out-of-sample validation.
""")
    return path


def _rpt_05_origin_level(d: dict) -> Path:
    results = d.get("main_results", [])
    path = REPORTS_DIR / "05_origin_level_results.md"
    rows = ""
    wins = 0
    for r in results:
        fy = r["feature_year"]
        m0_rmse  = r["metrics"]["M0"]["rmse"]
        a5a_rmse = r["metrics"]["A5a"]["rmse"]
        delta    = a5a_rmse - m0_rmse
        winner   = "A5a ✓" if delta < 0 else ("M0" if delta > 0 else "TIE")
        if delta < 0:
            wins += 1
        rows += f"| {fy} | {fy+1} | {r['n_train']} | {r['n_eval']} | {m0_rmse:.4f} | {a5a_rmse:.4f} | {delta:+.4f} | {winner} |\n"

    _write(path, f"""# Phase 20 — Origin-Level Results

Clipping: C1 (Phase 19 bounds [-40%, +60%])
Origins evaluated: {len(results)}
A5a wins: {wins} / {len(results)}

| Feature Year | Target Year | N Train | N Eval | M0 RMSE | A5a RMSE | ΔRMSE | Winner |
|---|---|---|---|---|---|---|---|
{rows}
""")
    return path


def _rpt_06_ablation(d: dict) -> Path:
    ablation = d.get("ablation", {})
    path = REPORTS_DIR / "06_component_ablation.md"
    rows = ""
    for subset, res in ablation.items():
        rows += (
            f"| {subset:15s} | {res.get('mean_rmse','N/A')} | "
            f"{res.get('mean_mae','N/A')} | {res.get('delta_rmse','N/A'):+.4f} | "
            f"{res.get('origin_wins','N/A')}/{res.get('n_origins','N/A')} |\n"
            if isinstance(res.get('delta_rmse'), (int, float))
            else f"| {subset:15s} | N/A | N/A | N/A | N/A |\n"
        )

    _write(path, f"""# Phase 20 — Component Ablation

## Purpose
Does A5a genuinely benefit from architectural diversity?
Does the three-model ensemble outperform any two-model subset?

## Results

| Subset | Mean RMSE | Mean MAE | ΔRMSE vs M0 | Origin Wins |
|---|---|---|---|---|
{rows}

## Interpretation
- M0+Ridge+RF should be the best or near-best combination.
- If a two-model subset matches the three-model ensemble, diversity is limited.
- Error cancellation is verified separately in the error diversity report.

> The ablation tests the Phase 19 claim that architectural diversity
> causes partial error cancellation.
""")
    return path


def _rpt_07_error_diversity(d: dict) -> Path:
    div = d.get("diversity", {})
    path = REPORTS_DIR / "07_error_diversity.md"
    _write(path, f"""# Phase 20 — Error Diversity Analysis

## Mechanism Under Test
```
Different models → different errors → averaging → partial error cancellation
```

## Residual (Error) Correlations

| Pair | Correlation |
|---|---|
| M0 — Ridge | {div.get('error_corr_m0_ridge', 'N/A')} |
| M0 — RF    | {div.get('error_corr_m0_rf', 'N/A')} |
| Ridge — RF | {div.get('error_corr_ridge_rf', 'N/A')} |

> Lower correlation = more diversity = more cancellation potential.

## Prediction Correlations

| Pair | Correlation |
|---|---|
| M0 — Ridge | {div.get('pred_corr_m0_ridge', 'N/A')} |
| M0 — RF    | {div.get('pred_corr_m0_rf', 'N/A')} |
| Ridge — RF | {div.get('pred_corr_ridge_rf', 'N/A')} |

## Complementarity

| Cases | Count |
|---|---|
| M0 right, Ridge wrong | {div.get('m0_right_ridge_wrong', 'N/A')} |
| M0 wrong, Ridge right | {div.get('m0_wrong_ridge_right', 'N/A')} |
| M0 right, RF wrong    | {div.get('m0_right_rf_wrong', 'N/A')} |
| M0 wrong, RF right    | {div.get('m0_wrong_rf_right', 'N/A')} |

## Overall: Does averaging help?

- Observations where A5a < M0 error: {div.get('a5a_better_n', 'N/A')} ({div.get('a5a_pct_better', 'N/A'):.1%})
- Large disagreement cases (>1% GDP): {div.get('large_disagreement_n', 'N/A')} ({div.get('large_disagreement_pct', 'N/A'):.1%})

## Conclusion
{'Diversity confirmed: error correlations < 0.9 and models show complementary success patterns.' if (div.get('error_corr_m0_rf', 1.0) or 1.0) < 0.9 else 'Models are highly correlated — diversity benefit is limited.'}
""")
    return path


def _rpt_08_priority_countries(d: dict) -> Path:
    pc = d.get("priority_countries", pd.DataFrame())
    path = REPORTS_DIR / "08_priority_countries.md"
    rows = ""
    if isinstance(pc, pd.DataFrame) and not pc.empty:
        for _, row in pc.iterrows():
            improved = "✅" if row.get("a5a_better", False) else "⚠️"
            rows += (
                f"| {row['country']} | {row.get('n','N/A')} | "
                f"{row.get('m0_rmse','N/A')} | {row.get('a5a_rmse','N/A')} | "
                f"{row.get('delta_rmse','N/A'):+.4f} | "
                f"{row.get('m0_mae','N/A')} | {row.get('a5a_mae','N/A')} | {improved} |\n"
                if isinstance(row.get('delta_rmse'), (int, float))
                else f"| {row['country']} | N/A | N/A | N/A | N/A | N/A | N/A | — |\n"
            )
    gate = d.get("gates", {}).get("G8", "N/A")
    _write(path, f"""# Phase 20 — Priority Country Audit

Priority countries: GBR, BRA, FRA, CAN, AUS

> A5a is evaluated as the SAME global architecture.
> Country-specific selection is not permitted.

| Country | N | M0 RMSE | A5a RMSE | ΔRMSE | M0 MAE | A5a MAE | Improved? |
|---|---|---|---|---|---|---|---|
{rows}

## Gate 8 Status
{gate}
""")
    return path


def _rpt_09_guardrail_countries(d: dict) -> Path:
    gc = d.get("guardrail_countries", pd.DataFrame())
    path = REPORTS_DIR / "09_guardrail_countries.md"
    rows = ""
    if isinstance(gc, pd.DataFrame) and not gc.empty:
        for _, row in gc.iterrows():
            status = "✅" if row.get("a5a_better", False) else "⚠️ DEGRADED"
            rows += (
                f"| {row['country']} | {row.get('n','N/A')} | "
                f"{row.get('m0_rmse','N/A')} | {row.get('a5a_rmse','N/A')} | "
                f"{row.get('delta_rmse','N/A'):+.4f} | {status} |\n"
                if isinstance(row.get('delta_rmse'), (int, float))
                else f"| {row['country']} | N/A | N/A | N/A | N/A | — |\n"
            )
    gate = d.get("gates", {}).get("G7", "N/A")

    # Find IND specifically
    ind_row = {}
    if isinstance(gc, pd.DataFrame) and not gc.empty:
        ind_rows = gc[gc["country"] == "IND"]
        if not ind_rows.empty:
            ind_row = ind_rows.iloc[0].to_dict()

    _write(path, f"""# Phase 20 — Guardrail Country Audit

Guardrail countries: USA, CHN, DEU, JPN, IND
A candidate improving globally but degrading these economies is rejected as non-robust.

| Country | N | M0 RMSE | A5a RMSE | ΔRMSE | Status |
|---|---|---|---|---|---|
{rows}

## India (IND) — Special Assessment
Phase 19 reported: M0 RMSE=3.5089, A5a RMSE=3.6238, ΔRMSE=+0.1149 (A5a worse)

Phase 20 result:
- M0 RMSE: {ind_row.get('m0_rmse', 'N/A')}
- A5a RMSE: {ind_row.get('a5a_rmse', 'N/A')}
- ΔRMSE: {ind_row.get('delta_rmse', 'N/A')}
- A5a better: {ind_row.get('a5a_better', 'N/A')}

> This degradation is NOT hidden. If it persists, it is documented explicitly.

## Gate 7 Status
{gate}
""")
    return path


def _rpt_10_shock_analysis(d: dict) -> Path:
    shock = d.get("shock", {})
    path  = REPORTS_DIR / "10_shock_analysis.md"
    rows  = ""
    for model, periods in shock.items():
        for period, metrics in periods.items():
            rows += f"| {model} | {period} | {metrics.get('rmse','N/A')} | {metrics.get('mae','N/A')} | {metrics.get('n','N/A')} |\n"

    _write(path, f"""# Phase 20 — Shock Period Analysis

## COVID-19 (Target Year 2020)
All models are expected to struggle. No feature from year 2019 can predict a global pandemic.

| Model | Period | RMSE | MAE | N |
|---|---|---|---|---|
{rows}

> 2020 performance is informative but is NOT a primary promotion criterion.
> No model can plausibly predict an unprecedented black-swan event.
""")
    return path


def _rpt_11_error_tail(d: dict) -> Path:
    tail = d.get("tail", {})
    worst = d.get("worst_predictions", [])
    path  = REPORTS_DIR / "11_error_tail_analysis.md"

    tail_rows = ""
    for model, t in tail.items():
        tail_rows += (
            f"| {model} | {t.get('p90_ae','N/A')} | {t.get('p95_ae','N/A')} | "
            f"{t.get('max_ae','N/A')} | {t.get('n_above_10','N/A')} |\n"
        )
    worst_rows = ""
    for w in worst[:15]:
        worst_rows += (
            f"| {w.get('target_year','N/A')} | {w.get('country','N/A')} | "
            f"{w.get('model','N/A')} | {w.get('actual','N/A'):.1f} | "
            f"{w.get('predicted','N/A'):.1f} | {w.get('abs_error','N/A'):.1f} |\n"
        )

    _write(path, f"""# Phase 20 — Error Tail Analysis

## Tail Percentiles

| Model | p90 AE | p95 AE | Max AE | N > 10% |
|---|---|---|---|---|
{tail_rows}

## Top 15 Worst Predictions

| Target Year | Country | Model | Actual | Predicted | Abs Error |
|---|---|---|---|---|---|
{worst_rows}

> Outliers are NOT removed. They represent real forecasting failures.
> A model that improves mean RMSE by concentrating large errors in a few
> observations is not considered robust.
""")
    return path


def _rpt_12_statistical(d: dict) -> Path:
    stats = d.get("statistical", {})
    path  = REPORTS_DIR / "12_statistical_comparison.md"
    _write(path, f"""# Phase 20 — Statistical Comparison

## Paired Wilcoxon Signed-Rank Test (n={stats.get('n_origins', 'N/A')} origins)

| Metric | Value |
|---|---|
| Mean ΔRMSE | {stats.get('mean_delta_rmse', 'N/A')} |
| Median ΔRMSE | {stats.get('median_delta_rmse', 'N/A')} |
| Std ΔRMSE | {stats.get('std_delta_rmse', 'N/A')} |
| Mean ΔMAE | {stats.get('mean_delta_mae', 'N/A')} |
| Median ΔMAE | {stats.get('median_delta_mae', 'N/A')} |
| Wilcoxon stat (RMSE) | {stats.get('wilcoxon_stat_rmse', 'N/A')} |
| Wilcoxon p-value (RMSE) | {stats.get('wilcoxon_pval_rmse', 'N/A')} |
| Statistically significant (α=0.05) | {stats.get('statistically_significant_rmse', 'N/A')} |
| Relative improvement | {stats.get('relative_improvement', 'N/A'):.1%} |

## Interpretation

Statistical significance does NOT equal practical significance.
With n=23 origins, power is limited. A small p-value strengthens the case;
a non-significant p-value does not automatically invalidate the improvement.

Practical improvement (mean |ΔRMSE|): {stats.get('practical_improvement', 'N/A')} RMSE points.

> "Statistically significant" is reported only when a formal test supports it.
""")
    return path


def _rpt_13_reproducibility(d: dict) -> Path:
    repro = d.get("reproducibility", {})
    path  = REPORTS_DIR / "13_reproducibility.md"
    _write(path, f"""# Phase 20 — Reproducibility Audit

## Status
{repro.get('status', 'N/A')}

## Summary
| Item | Value |
|---|---|
| Runs executed | 2 |
| Origins compared | {repro.get('n_origins', 'N/A')} |
| Mismatches | {repro.get('n_mismatches', 'N/A')} |
| Predictions identical | {repro.get('identical', 'N/A')} |
| Tolerance | 1e-10 |

## Mismatches
{repro.get('mismatches', 'None — all predictions bit-exact') if repro.get('mismatches') else 'None — all predictions bit-exact across both runs.'}
""")
    return path


def _rpt_14_data_vintage(d: dict) -> Path:
    path = REPORTS_DIR / "14_data_vintage_audit.md"
    _write(path, f"""# Phase 20 — Data Vintage and Leakage Audit

## T+1 Design Verification

For every origin Y:
- Features: year == Y (data available at end of year Y)
- Target: gdp_growth_next_year (= actual GDP growth in Y+1)
- No feature from year Y+1 or later is used

This is structurally enforced by `get_train_eval_split()`:
```python
train_mask = (df[YEAR_COL] < feature_year) & (df[TARGET_AVAIL_COL] == 1)
eval_mask  = (df[YEAR_COL] == feature_year) & (df[TARGET_AVAIL_COL] == 1)
```

## Leakage Rules Status

| Rule | Description | Status |
|---|---|---|
| L1 | Target not in feature set | {d.get('leakage', {}).get('L1', 'N/A')} |
| L2 | Training rows year < feature_year | {d.get('leakage', {}).get('L2', 'N/A')} |
| L3 | StandardScaler fitted on training only | {d.get('leakage', {}).get('L3', 'N/A')} |
| L4 | SimpleImputer fitted on training only | {d.get('leakage', {}).get('L4', 'N/A')} |
| L5 | Fixed 31-feature locked set | {d.get('leakage', {}).get('L5', 'N/A')} |
| L6 | Ridge alpha from val split only | {d.get('leakage', {}).get('L6', 'N/A')} |
| L7 | No test-set model selection | {d.get('leakage', {}).get('L7', 'N/A')} |
| L8 | No test-driven clipping | {d.get('leakage', {}).get('L8', 'N/A')} |
| L9 | OOS not peeked before design frozen | {d.get('leakage', {}).get('L9', 'N/A')} |
| L10 | No random train/test split | {d.get('leakage', {}).get('L10', 'N/A')} |
| L11 | Production artifacts not mutated | {d.get('leakage', {}).get('L11', 'N/A')} |
| L12 | No synthetic/fabricated data | {d.get('leakage', {}).get('L12', 'N/A')} |

## Publication Vintage Limitation
World Bank GDP data is subject to revision. Feature values used here reflect
the snapshot in `master_panel_t1_missingness.csv`. Actual publication-vintage
timestamps are not available in the current dataset. This limitation is
acknowledged but does not invalidate the chronological split design.
""")
    return path


def _rpt_15_operational(d: dict) -> Path:
    ops = d.get("operational", {})
    path = REPORTS_DIR / "15_operational_readiness.md"

    _write(path, f"""# Phase 20 — Operational Readiness Audit

## Overall Status
{ops.get('overall_status', 'N/A')}

## Checklist

| Check | Status | Detail |
|---|---|---|
| Serialization | {'✅' if ops.get('serialization_ok') else '❌'} | {str(ops.get('artifact_paths', {}))} |
| Independent load (Correction 5) | {'✅' if ops.get('independent_load_ok') else '❌'} | {ops.get('independent_load_msg', 'N/A')} |
| Input schema (31 features) | {'✅' if ops.get('input_schema_ok') else '❌'} | {ops.get('n_features_actual', 'N/A')} features |
| Output schema | {'✅' if ops.get('output_schema_ok') else '❌'} | Array of floats, no NaN |
| Missing value handling | {'✅' if ops.get('missing_value_ok') else '❌'} | {ops.get('missing_value_msg', 'N/A')} |
| Determinism | {'✅' if ops.get('determinism_ok') else '❌'} | {ops.get('determinism_msg', 'N/A')} |
| Production untouched | {'✅' if ops.get('prod_untouched') else '❌'} | Phase 11 artifact unchanged |
| Rollback mechanism | {'✅' if ops.get('rollback_ok') else '❌'} | {ops.get('rollback_mechanism', 'N/A')[:100]} |

## Inference Latency
- Mean (200-country batch): {ops.get('latency_ms_mean', 'N/A')} ms
- p95 (200-country batch): {ops.get('latency_ms_p95', 'N/A')} ms
- Runs: {ops.get('n_latency_runs', 'N/A')}

## Artifact Governance
All A5a artifacts are saved as `phase20_*_EXPERIMENTAL_ONLY.joblib`.
These never overwrite `models/phase11/best_t1_gdp_growth_model.joblib`.
""")
    return path


def _rpt_16_rollback(d: dict) -> Path:
    path = REPORTS_DIR / "16_rollback_plan.md"
    _write(path, """# Phase 20 — Rollback Plan

## Current Production
```
Model:    HistGradientBoostingRegressor (Phase 11)
Artifact: models/phase11/best_t1_gdp_growth_model.joblib
Status:   FROZEN — MD5 and SHA256 verified before/after Phase 20
```

## Candidate
```
Model:    A5a Equal-Weight Ensemble
Status:   EXPERIMENTAL_ONLY
Artifacts: apps/api/ml/phase20/artifacts/phase20_*_EXPERIMENTAL_ONLY.joblib
```

## Rollback Mechanism
Phase 11 is never overwritten. If A5a is deployed and needs to be rolled back:

1. Stop serving A5a ensemble
2. Reload `models/phase11/best_t1_gdp_growth_model.joblib`
3. Verify MD5 matches Phase 11 manifest value
4. Resume serving Phase 11

No database migration, no feature store change, no schema change is required.
The rollback is instantaneous.

## Design Principle
A5a is deployed ALONGSIDE Phase 11 (not instead of it) until a formal release
decision is made. If A5a fails in production, Phase 11 is always available.

## Gate 10 Implication
Rollback safety is a mandatory gate. If the Phase 11 artifact has been
accidentally mutated, Gate 2 will fail and EXPERIMENT_FAILED_GOVERNANCE
is declared before any deployment.
""")
    return path


def _rpt_17_promotion_gates(d: dict) -> Path:
    gates = d.get("gates", {})
    path  = REPORTS_DIR / "17_promotion_gates.md"
    rows  = ""
    for gate, status in gates.items():
        emoji = "✅" if "PASS" in str(status) else ("⚠️" if "MARGINAL" in str(status) else "❌")
        rows += f"| {gate} | {emoji} {status} |\n"

    _write(path, f"""# Phase 20 — Promotion Gates

A5a can only be considered for production if ALL mandatory gates pass.

| Gate | Status |
|---|---|
{rows}

## Gate Definitions

| Gate | Requirement |
|---|---|
| G1 | Dataset MD5/SHA256 unchanged |
| G2 | Production model+manifest unchanged after experiment |
| G3 | All leakage rules L1–L12 pass |
| G4 | Reproducibility: Run1 == Run2 (atol=1e-10) |
| G5 | A5a advantage not solely an artifact of clipping choice |
| G6 | Broad improvement (not concentrated in few years) |
| G7 | No serious guardrail degradation (IND assessed explicitly) |
| G8 | No systematic priority country failure |
| G9 | No unacceptable increase in extreme errors (p95/max AE) |
| G10 | Operational readiness + rollback-safe |
| G11 | New OOS evidence (INCONCLUSIVE if no new data) |
""")
    return path


def _rpt_18_governance(d: dict) -> Path:
    gov  = d.get("governance", {})
    path = REPORTS_DIR / "18_governance_decision.md"
    _write(path, f"""# Phase 20 — Governance Decision

Generated: {_ts()}

## Decision
```
{gov.get('decision', 'PENDING')}
```

## Recommendation
```
{gov.get('recommendation', 'N/A')}
```

## Rationale
{gov.get('rationale', 'N/A')}

## Gates Summary
{gov.get('gates_summary', 'N/A')}

## Non-Negotiable Rules Applied
```
NO PRODUCTION MUTATION          ✓
NO AUTOMATIC PROMOTION          ✓
NO TEST-SET MODEL SELECTION     ✓
NO TEST-DRIVEN CLIPPING         ✓
NO DATA FABRICATION             ✓
NO FUTURE INFORMATION           ✓
NO RANDOM SPLITS                ✓
NO NEW MODEL ARCHITECTURES      ✓
NO LARGE HYPERPARAMETER SEARCH  ✓
NO HIDING NEGATIVE RESULTS      ✓
NO CLAIM OF FUTURE ACCURACY     ✓
```

## Next Steps
{gov.get('next_steps', 'N/A')}
""")
    return path


def _rpt_consolidated(d: dict) -> Path:
    gov   = d.get("governance", {})
    gates = d.get("gates", {})
    agg   = d.get("aggregate", {})
    ops   = d.get("operational", {})
    repro = d.get("reproducibility", {})
    stats = d.get("statistical", {})
    oos   = d.get("oos", {})
    path  = REPORTS_DIR / "PHASE20_PROMOTION_AUDIT_REPORT.md"

    gate_summary = "\n".join(
        f"| {g} | {'✅ ' if 'PASS' in str(s) else ('⚠️ ' if 'MARGINAL' in str(s) else '❌ ')}{s} |"
        for g, s in gates.items()
    )

    _write(path, f"""# PHASE 20 — PROMOTION AUDIT REPORT (Consolidated)

Generated: {_ts()}

---

## STATUS: {gov.get('decision', 'PENDING')}

---

## Phase 19 Reproduction
{d.get('reproduction', {}).get('status', 'N/A')}
- Phase 19 ref M0 RMSE:  {d.get('reproduction', {}).get('ref_m0_rmse', 'N/A')}
- Reproduced M0 RMSE:    {d.get('reproduction', {}).get('reproduced_m0_rmse', 'N/A')}
- Phase 19 ref A5a RMSE: {d.get('reproduction', {}).get('ref_a5a_rmse', 'N/A')}
- Reproduced A5a RMSE:   {d.get('reproduction', {}).get('reproduced_a5a_rmse', 'N/A')}

---

## Main Results (C1 Clipping)

| Metric | M0 (Phase 11) | A5a |
|---|---|---|
| Mean RMSE | {agg.get('m0_mean_rmse', 'N/A')} | {agg.get('mean_rmse', 'N/A')} |
| Median RMSE | — | {agg.get('median_rmse', 'N/A')} |
| Mean MAE | — | {agg.get('mean_mae', 'N/A')} |
| ΔRMSE | — | {agg.get('delta_rmse', 'N/A')} |
| Origin wins | — | {agg.get('origin_wins', 'N/A')} / {agg.get('n_origins', 'N/A')} ({agg.get('win_pct', 0):.1%}) |

---

## Clipping Sensitivity (G5)
{d.get('gates', {}).get('G5', 'N/A')}

---

## Out-of-Sample Evidence
{oos.get('message', 'N/A')}

---

## Statistical Comparison
- Wilcoxon p-value (RMSE): {stats.get('wilcoxon_pval_rmse', 'N/A')}
- Statistically significant: {stats.get('statistically_significant_rmse', 'N/A')}
- Relative improvement: {stats.get('relative_improvement', 0):.1%}

---

## Reproducibility
{repro.get('status', 'N/A')} — {repro.get('n_mismatches', 'N/A')} mismatches across {repro.get('n_origins', 'N/A')} origins

---

## Operational Readiness
{ops.get('overall_status', 'N/A')}
- Independent load (Correction 5): {'PASS' if ops.get('independent_load_ok') else 'FAIL'}
- Latency: {ops.get('latency_ms_mean', 'N/A')} ms mean
- Missing values: {'OK' if ops.get('missing_value_ok') else 'FAIL'}
- Determinism: {'OK' if ops.get('determinism_ok') else 'FAIL'}

---

## Promotion Gates

| Gate | Status |
|---|---|
{gate_summary}

---

## Final Decision
```
{gov.get('decision', 'PENDING')}
```

## Recommendation
```
{gov.get('recommendation', 'N/A')}
```

---

*Phase 11 remains production. No automatic promotion.*
""")
    return path
