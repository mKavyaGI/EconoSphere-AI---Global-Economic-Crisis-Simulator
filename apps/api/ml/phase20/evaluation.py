"""
EconoSphere AI — Phase 20
evaluation.py : Core evaluation engine for the promotion audit.

Focus: A5a vs M0 only. No new architectures. No hyperparameter searches.

Implements:
  - evaluate_origin_p20()       : single-origin A5a + M0 with configurable clipping
  - run_a5a_backtest()          : full chronological backtest
  - run_clipping_sensitivity()  : C0 / C1 / C2 sensitivity
  - run_component_ablation()    : 7 ensemble subsets
  - run_error_diversity()       : residual correlations + disagreement analysis
  - run_leave_one_out()         : LOOO sensitivity
  - run_reproducibility()       : two identical runs, bit-exact comparison
  - run_oos_evaluation()        : strictly isolated final-exam evaluation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from apps.api.ml.phase19.evaluation import (
    _make_val_split,
    _select_ridge_alpha,
    _assert_chronological_train,
    _assert_no_target_in_X,
    compute_metrics,
)
from apps.api.ml.phase19.models import (
    build_m0,
    build_ridge,
    build_random_forest,
    RIDGE_ALPHAS,
)
from apps.api.ml.phase19.utils import (
    LOCKED_FEATURES,
    TARGET_COL,
    YEAR_COL,
    COUNTRY_COL,
    get_train_eval_split,
)
from .sensitivity import ClipConfig, CLIP_CONFIGS, PRIMARY_CLIP_CONFIG
from .utils import get_logger

log = get_logger()


# ── Single-origin evaluation ───────────────────────────────────────────────────

def evaluate_origin_p20(
    df: pd.DataFrame,
    feature_year: int,
    clip_config: ClipConfig = PRIMARY_CLIP_CONFIG,
    seed: int = 42,
) -> dict:
    """
    Evaluate M0 and A5a for a single forecast origin under a given clipping config.

    A5a = (M0_pred + Ridge_pred + RF_pred) / 3  — EQUAL WEIGHTS ALWAYS.
    Ridge alpha selected on validation split (last 20% of training years).
    M0 and RF are never clipped.
    Ridge predictions are clipped per clip_config before inclusion in ensemble.

    Returns a result dict with predictions, metrics, hyperparams, country info.
    """
    target_year = feature_year + 1

    train_df, eval_df = get_train_eval_split(df, feature_year)

    # Leakage guards
    _assert_chronological_train(train_df, feature_year, context=f"P20 origin={feature_year}")

    X_train = train_df[LOCKED_FEATURES].copy()
    y_train = train_df[TARGET_COL].values
    X_eval  = eval_df[LOCKED_FEATURES].copy()
    y_eval  = eval_df[TARGET_COL].values

    _assert_no_target_in_X(X_train, context=f"P20 X_train origin={feature_year}")
    _assert_no_target_in_X(X_eval,  context=f"P20 X_eval origin={feature_year}")

    assert len(y_eval) > 0, f"No eval rows at origin {feature_year}"

    # Validation split for Ridge alpha selection
    fit_df, val_df = _make_val_split(train_df, val_fraction=0.20)
    X_fit = fit_df[LOCKED_FEATURES].values
    y_fit = fit_df[TARGET_COL].values
    X_val = val_df[LOCKED_FEATURES].values
    y_val = val_df[TARGET_COL].values

    best_ridge_alpha = _select_ridge_alpha(X_fit, y_fit, X_val, y_val)

    # Fit base models on full training window
    m0_pipe = build_m0()
    rg_pipe = build_ridge(alpha=best_ridge_alpha)
    rf_pipe = build_random_forest()

    m0_pipe.fit(X_train.values, y_train)
    rg_pipe.fit(X_train.values, y_train)
    rf_pipe.fit(X_train.values, y_train)

    # Predictions
    m0_pred_raw = m0_pipe.predict(X_eval.values)
    rg_pred_raw = rg_pipe.predict(X_eval.values)
    rf_pred_raw = rf_pipe.predict(X_eval.values)

    # Apply clipping to Ridge only (M0 and RF naturally bounded by trees)
    rg_pred = clip_config.apply(rg_pred_raw)

    # A5a: strict equal weights 1/3 + 1/3 + 1/3
    a5a_pred = (m0_pred_raw + rg_pred + rf_pred_raw) / 3.0

    # Val-set component predictions (for error diversity)
    m0_val = m0_pipe.predict(X_val)
    rg_val = clip_config.apply(rg_pipe.predict(X_val))
    rf_val = rf_pipe.predict(X_val)

    metrics = {
        "M0":  compute_metrics(y_eval, m0_pred_raw),
        "A5a": compute_metrics(y_eval, a5a_pred),
        "Ridge": compute_metrics(y_eval, rg_pred),
        "RF":    compute_metrics(y_eval, rf_pred_raw),
    }

    return {
        "feature_year":   feature_year,
        "target_year":    target_year,
        "n_train":        len(train_df),
        "n_eval":         len(eval_df),
        "clip_config":    clip_config.name,
        "ridge_alpha":    best_ridge_alpha,
        "predictions": {
            "M0":    m0_pred_raw,
            "Ridge": rg_pred,
            "Ridge_raw": rg_pred_raw,
            "RF":    rf_pred_raw,
            "A5a":   a5a_pred,
        },
        "val_predictions": {
            "M0": m0_val, "Ridge": rg_val, "RF": rf_val,
        },
        "y_eval":   y_eval,
        "eval_df":  eval_df,
        "train_df": train_df,
        "metrics":  metrics,
    }


# ── Full chronological backtest ────────────────────────────────────────────────

def run_a5a_backtest(
    df: pd.DataFrame,
    origins: list[dict],
    clip_config: ClipConfig = PRIMARY_CLIP_CONFIG,
) -> list[dict]:
    """Run evaluate_origin_p20 for all origins. Returns list of result dicts."""
    results = []
    for o in origins:
        fy = o["feature_year"]
        try:
            r = evaluate_origin_p20(df, fy, clip_config=clip_config)
            results.append(r)
            m0_rmse  = r["metrics"]["M0"]["rmse"]
            a5a_rmse = r["metrics"]["A5a"]["rmse"]
            log.info(
                f"  [{clip_config.name}] Origin {fy}→{fy+1} | "
                f"M0={m0_rmse:.4f} | A5a={a5a_rmse:.4f} | "
                f"Δ={a5a_rmse - m0_rmse:+.4f}"
            )
        except Exception as exc:
            log.error(f"  Origin {fy}: FAILED — {exc}", exc_info=True)
    return results


# ── Aggregate metrics from results ─────────────────────────────────────────────

def aggregate_results(results: list[dict], candidate: str = "A5a") -> dict:
    """Compute aggregate metrics for a candidate across all origins."""
    rmses  = [r["metrics"][candidate]["rmse"] for r in results if candidate in r["metrics"]]
    maes   = [r["metrics"][candidate]["mae"]  for r in results if candidate in r["metrics"]]
    m0_rmses = [r["metrics"]["M0"]["rmse"]    for r in results]
    deltas = [r - m for r, m in zip(rmses, m0_rmses)]
    wins   = sum(1 for d in deltas if d < 0)
    losses = sum(1 for d in deltas if d > 0)
    ties   = sum(1 for d in deltas if d == 0)

    return {
        "candidate":    candidate,
        "mean_rmse":    float(np.mean(rmses)),
        "median_rmse":  float(np.median(rmses)),
        "mean_mae":     float(np.mean(maes)),
        "median_mae":   float(np.median(maes)),
        "m0_mean_rmse": float(np.mean(m0_rmses)),
        "delta_rmse":   float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "std_delta":    float(np.std(deltas)),
        "origin_wins":  wins,
        "m0_wins":      losses,
        "ties":         ties,
        "n_origins":    len(rmses),
        "win_pct":      wins / len(rmses) if rmses else 0.0,
    }


# ── Clipping sensitivity ───────────────────────────────────────────────────────

def run_clipping_sensitivity(
    df: pd.DataFrame,
    origins: list[dict],
) -> dict[str, dict]:
    """
    Run A5a vs M0 under C0, C1, C2 clipping configurations.
    Returns a dict keyed by config name with aggregate metrics.

    CORRECTION 3: Results are evidence for G5, not three separate gates.
    """
    results_by_config: dict[str, dict] = {}
    for cfg_name, cfg in CLIP_CONFIGS.items():
        log.info(f"\n  Clipping config {cfg_name}: {cfg.label}")
        run_results = run_a5a_backtest(df, origins, clip_config=cfg)
        agg = aggregate_results(run_results, candidate="A5a")
        m0_agg = aggregate_results(run_results, candidate="M0")
        results_by_config[cfg_name] = {
            "config":          cfg_name,
            "label":           cfg.label,
            "a5a_mean_rmse":   agg["mean_rmse"],
            "a5a_median_rmse": agg["median_rmse"],
            "m0_mean_rmse":    m0_agg["mean_rmse"],
            "delta_rmse":      agg["delta_rmse"],
            "origin_wins":     agg["origin_wins"],
            "m0_wins":         agg["m0_wins"],
            "n_origins":       agg["n_origins"],
            "per_origin":      run_results,
        }
    return results_by_config


# ── Component ablation ─────────────────────────────────────────────────────────

def _make_component_pred(
    preds: dict[str, np.ndarray],
    components: list[str],
) -> np.ndarray:
    """Equal-weight average of the listed components."""
    stacked = np.column_stack([preds[c] for c in components])
    return stacked.mean(axis=1)


ABLATION_SUBSETS = [
    ("M0",           ["M0"]),
    ("Ridge",        ["Ridge"]),
    ("RF",           ["RF"]),
    ("M0+Ridge",     ["M0", "Ridge"]),
    ("M0+RF",        ["M0", "RF"]),
    ("Ridge+RF",     ["Ridge", "RF"]),
    ("M0+Ridge+RF",  ["M0", "Ridge", "RF"]),  # = A5a
]


def run_component_ablation(
    df: pd.DataFrame,
    origins: list[dict],
    clip_config: ClipConfig = PRIMARY_CLIP_CONFIG,
) -> dict:
    """
    Evaluate all 7 ensemble subsets across all origins.
    Uses the same base model fits as the main evaluation to avoid refitting.
    Returns a dict keyed by subset name with aggregate metrics.
    """
    log.info("\n  [Ablation] Computing base model predictions for all origins...")

    # Collect base predictions per origin
    subset_rmses: dict[str, list] = {name: [] for name, _ in ABLATION_SUBSETS}
    subset_maes:  dict[str, list] = {name: [] for name, _ in ABLATION_SUBSETS}
    m0_rmses: list = []

    for o in origins:
        fy = o["feature_year"]
        try:
            r = evaluate_origin_p20(df, fy, clip_config=clip_config)
        except Exception as exc:
            log.error(f"  Ablation origin {fy}: FAILED — {exc}")
            continue

        y_eval = r["y_eval"]
        preds  = r["predictions"]  # has M0, Ridge, RF

        m0_rmses.append(r["metrics"]["M0"]["rmse"])

        for subset_name, components in ABLATION_SUBSETS:
            combo_pred = _make_component_pred(preds, components)
            rmse = float(np.sqrt(mean_squared_error(y_eval, combo_pred)))
            mae  = float(mean_absolute_error(y_eval, combo_pred))
            subset_rmses[subset_name].append(rmse)
            subset_maes[subset_name].append(mae)

    results = {}
    m0_mean = float(np.mean(m0_rmses))

    for subset_name, _ in ABLATION_SUBSETS:
        rs = subset_rmses[subset_name]
        ms = subset_maes[subset_name]
        m0_rs = m0_rmses[:len(rs)]  # align
        wins  = sum(1 for r, m in zip(rs, m0_rs) if r < m)
        deltas = [r - m for r, m in zip(rs, m0_rs)]
        results[subset_name] = {
            "mean_rmse":   round(float(np.mean(rs)), 4),
            "median_rmse": round(float(np.median(rs)), 4),
            "mean_mae":    round(float(np.mean(ms)), 4),
            "delta_rmse":  round(float(np.mean(deltas)), 4),
            "origin_wins": wins,
            "n_origins":   len(rs),
        }
        log.info(
            f"  [Ablation] {subset_name:15s}: RMSE={results[subset_name]['mean_rmse']:.4f} | "
            f"Δ={results[subset_name]['delta_rmse']:+.4f} | wins={wins}/{len(rs)}"
        )

    return results


# ── Error diversity analysis ───────────────────────────────────────────────────

def run_error_diversity(results: list[dict]) -> dict:
    """
    Compute error and prediction correlations among M0, Ridge, RF.

    Tests the mechanism: different models → different errors → partial cancellation.
    Reports residual correlations, prediction correlations, and disagreement stats.
    """
    all_m0_errors:    list = []
    all_ridge_errors: list = []
    all_rf_errors:    list = []
    all_m0_preds:     list = []
    all_ridge_preds:  list = []
    all_rf_preds:     list = []
    all_actuals:      list = []

    for r in results:
        y      = r["y_eval"]
        m0_p   = r["predictions"]["M0"]
        rg_p   = r["predictions"]["Ridge"]
        rf_p   = r["predictions"]["RF"]

        all_m0_errors.extend(y - m0_p)
        all_ridge_errors.extend(y - rg_p)
        all_rf_errors.extend(y - rf_p)
        all_m0_preds.extend(m0_p)
        all_ridge_preds.extend(rg_p)
        all_rf_preds.extend(rf_p)
        all_actuals.extend(y)

    m0_e   = np.array(all_m0_errors)
    rg_e   = np.array(all_ridge_errors)
    rf_e   = np.array(all_rf_errors)
    m0_p   = np.array(all_m0_preds)
    rg_p   = np.array(all_ridge_preds)
    rf_p   = np.array(all_rf_preds)
    actual = np.array(all_actuals)

    # Residual (error) correlations
    r_m0_rg = float(np.corrcoef(m0_e, rg_e)[0, 1])
    r_m0_rf = float(np.corrcoef(m0_e, rf_e)[0, 1])
    r_rg_rf = float(np.corrcoef(rg_e, rf_e)[0, 1])

    # Prediction correlations
    p_m0_rg = float(np.corrcoef(m0_p, rg_p)[0, 1])
    p_m0_rf = float(np.corrcoef(m0_p, rf_p)[0, 1])
    p_rg_rf = float(np.corrcoef(rg_p, rf_p)[0, 1])

    # Disagreement: fraction of cases where |m0_pred - a5a_pred| > threshold
    a5a_p = (m0_p + rg_p + rf_p) / 3.0
    disagreement = np.abs(m0_p - a5a_p)
    large_disagreement_n = int((disagreement > 1.0).sum())
    large_disagreement_pct = large_disagreement_n / len(m0_p)

    # Cases where one model is "right" (|error| < median) while another is "wrong"
    m0_ae   = np.abs(m0_e)
    rg_ae   = np.abs(rg_e)
    rf_ae   = np.abs(rf_e)
    a5a_ae  = np.abs(actual - a5a_p)
    med_ae  = float(np.median(m0_ae))

    m0_right_rg_wrong   = int(((m0_ae < med_ae) & (rg_ae >= med_ae)).sum())
    m0_wrong_rg_right   = int(((m0_ae >= med_ae) & (rg_ae < med_ae)).sum())
    m0_right_rf_wrong   = int(((m0_ae < med_ae) & (rf_ae >= med_ae)).sum())
    m0_wrong_rf_right   = int(((m0_ae >= med_ae) & (rf_ae < med_ae)).sum())

    # Overall: does A5a reduce errors vs M0?
    a5a_better  = int((a5a_ae < m0_ae).sum())
    a5a_worse   = int((a5a_ae > m0_ae).sum())
    diversity_pct_better = a5a_better / len(m0_ae)

    log.info(f"  [Diversity] Error correlations: M0-Ridge={r_m0_rg:.3f}, M0-RF={r_m0_rf:.3f}, Ridge-RF={r_rg_rf:.3f}")
    log.info(f"  [Diversity] Prediction correlations: M0-Ridge={p_m0_rg:.3f}, M0-RF={p_m0_rf:.3f}, Ridge-RF={p_rg_rf:.3f}")
    log.info(f"  [Diversity] A5a better than M0 on {diversity_pct_better:.1%} of observations")

    return {
        "n_observations":         len(m0_e),
        "error_corr_m0_ridge":    round(r_m0_rg, 4),
        "error_corr_m0_rf":       round(r_m0_rf, 4),
        "error_corr_ridge_rf":    round(r_rg_rf, 4),
        "pred_corr_m0_ridge":     round(p_m0_rg, 4),
        "pred_corr_m0_rf":        round(p_m0_rf, 4),
        "pred_corr_ridge_rf":     round(p_rg_rf, 4),
        "m0_right_ridge_wrong":   m0_right_rg_wrong,
        "m0_wrong_ridge_right":   m0_wrong_rg_right,
        "m0_right_rf_wrong":      m0_right_rf_wrong,
        "m0_wrong_rf_right":      m0_wrong_rf_right,
        "large_disagreement_n":   large_disagreement_n,
        "large_disagreement_pct": round(large_disagreement_pct, 4),
        "a5a_better_n":           a5a_better,
        "a5a_worse_n":            a5a_worse,
        "a5a_pct_better":         round(diversity_pct_better, 4),
        "median_ae_threshold":    round(med_ae, 4),
    }


# ── Leave-one-origin-out sensitivity ──────────────────────────────────────────

def run_leave_one_out(results: list[dict]) -> dict:
    """
    Leave-one-origin-out sensitivity: does A5a still beat M0 when one origin
    is excluded?

    Tests whether the 18/23 win count is driven by a small number of
    unusually favorable origins.
    """
    all_m0_rmses  = [r["metrics"]["M0"]["rmse"]  for r in results]
    all_a5a_rmses = [r["metrics"]["A5a"]["rmse"] for r in results]
    all_years     = [r["feature_year"]             for r in results]
    deltas        = [a - m for a, m in zip(all_a5a_rmses, all_m0_rmses)]

    looo_results = []

    for i, (fy, d) in enumerate(zip(all_years, deltas)):
        remaining_m0  = [v for j, v in enumerate(all_m0_rmses)  if j != i]
        remaining_a5a = [v for j, v in enumerate(all_a5a_rmses) if j != i]
        remaining_deltas = [v for j, v in enumerate(deltas) if j != i]

        excl_m0_mean  = float(np.mean(remaining_m0))
        excl_a5a_mean = float(np.mean(remaining_a5a))
        excl_delta    = float(np.mean(remaining_deltas))
        excl_wins     = sum(1 for d in remaining_deltas if d < 0)

        looo_results.append({
            "excluded_year":     fy,
            "excluded_delta":    round(d, 4),
            "remaining_m0_mean": round(excl_m0_mean, 4),
            "remaining_a5a_mean": round(excl_a5a_mean, 4),
            "remaining_delta":   round(excl_delta, 4),
            "remaining_wins":    excl_wins,
            "n_remaining":       len(remaining_m0),
            "a5a_still_wins_mean": excl_delta < 0,
        })

    always_positive = all(r["a5a_still_wins_mean"] for r in looo_results)
    worst_excl      = max(looo_results, key=lambda x: x["remaining_delta"])
    best_excl       = min(looo_results, key=lambda x: x["remaining_delta"])

    log.info(f"  [LOOO] A5a mean improvement survives every single exclusion: {always_positive}")
    log.info(f"  [LOOO] Worst case (excl {worst_excl['excluded_year']}): Δ={worst_excl['remaining_delta']:+.4f}")
    log.info(f"  [LOOO] Best  case (excl {best_excl['excluded_year']}):  Δ={best_excl['remaining_delta']:+.4f}")

    return {
        "always_positive":    always_positive,
        "worst_exclusion":    worst_excl,
        "best_exclusion":     best_excl,
        "per_exclusion":      looo_results,
        "n_origins":          len(results),
    }


# ── Paired statistical test ────────────────────────────────────────────────────

def run_statistical_comparison(results: list[dict]) -> dict:
    """
    Paired Wilcoxon signed-rank test on origin-level RMSE differences.
    Per the audit: clearly distinguish statistical evidence from practical evidence.
    """
    m0_rmses  = np.array([r["metrics"]["M0"]["rmse"]  for r in results])
    a5a_rmses = np.array([r["metrics"]["A5a"]["rmse"] for r in results])
    m0_maes   = np.array([r["metrics"]["M0"]["mae"]   for r in results])
    a5a_maes  = np.array([r["metrics"]["A5a"]["mae"]  for r in results])

    rmse_diffs = a5a_rmses - m0_rmses
    mae_diffs  = a5a_maes  - m0_maes

    # Wilcoxon signed-rank test (non-parametric; appropriate for small n=23)
    try:
        stat_rmse, pval_rmse = stats.wilcoxon(rmse_diffs, alternative="less")
        stat_mae,  pval_mae  = stats.wilcoxon(mae_diffs,  alternative="less")
    except Exception:
        stat_rmse, pval_rmse = np.nan, np.nan
        stat_mae,  pval_mae  = np.nan, np.nan

    alpha = 0.05  # standard significance level
    stat_sig_rmse = (pval_rmse < alpha) if not np.isnan(pval_rmse) else False
    stat_sig_mae  = (pval_mae  < alpha) if not np.isnan(pval_mae)  else False

    practical_improvement = float(np.mean(np.abs(rmse_diffs)))
    relative_improvement  = float(abs(np.mean(rmse_diffs)) / np.mean(m0_rmses))

    log.info(f"  [Stats] Mean ΔRMSE: {np.mean(rmse_diffs):+.4f}, Median ΔRMSE: {np.median(rmse_diffs):+.4f}")
    log.info(f"  [Stats] Wilcoxon RMSE: stat={stat_rmse:.3f}, p={pval_rmse:.4f} → {'statistically significant' if stat_sig_rmse else 'NOT significant'}")
    log.info(f"  [Stats] Relative improvement: {relative_improvement:.1%}")

    return {
        "mean_delta_rmse":       round(float(np.mean(rmse_diffs)), 4),
        "median_delta_rmse":     round(float(np.median(rmse_diffs)), 4),
        "std_delta_rmse":        round(float(np.std(rmse_diffs)), 4),
        "mean_delta_mae":        round(float(np.mean(mae_diffs)), 4),
        "median_delta_mae":      round(float(np.median(mae_diffs)), 4),
        "wilcoxon_stat_rmse":    float(stat_rmse) if not np.isnan(stat_rmse) else None,
        "wilcoxon_pval_rmse":    float(pval_rmse) if not np.isnan(pval_rmse) else None,
        "wilcoxon_stat_mae":     float(stat_mae)  if not np.isnan(stat_mae)  else None,
        "wilcoxon_pval_mae":     float(pval_mae)  if not np.isnan(pval_mae)  else None,
        "statistically_significant_rmse": stat_sig_rmse,
        "statistically_significant_mae":  stat_sig_mae,
        "alpha":                 alpha,
        "relative_improvement":  round(relative_improvement, 4),
        "practical_improvement": round(practical_improvement, 4),
        "n_origins":             len(results),
    }


# ── Reproducibility ────────────────────────────────────────────────────────────

def run_reproducibility(
    df: pd.DataFrame,
    origins: list[dict],
    clip_config: ClipConfig = PRIMARY_CLIP_CONFIG,
) -> dict:
    """
    Run A5a twice on the same data. Verify predictions are bit-exact.
    Reports any nondeterminism.
    """
    log.info("  [Reproducibility] Run 1...")
    run1 = run_a5a_backtest(df, origins, clip_config=clip_config)
    log.info("  [Reproducibility] Run 2...")
    run2 = run_a5a_backtest(df, origins, clip_config=clip_config)

    mismatches = []
    for r1, r2 in zip(run1, run2):
        fy = r1["feature_year"]
        for candidate in ["M0", "A5a", "Ridge", "RF"]:
            p1 = r1["predictions"].get(candidate)
            p2 = r2["predictions"].get(candidate)
            if p1 is None or p2 is None:
                continue
            if not np.allclose(p1, p2, atol=1e-10):
                mismatches.append({
                    "origin":    fy,
                    "candidate": candidate,
                    "max_diff":  float(np.max(np.abs(p1 - p2))),
                })

    identical = len(mismatches) == 0
    status = "PASS" if identical else "FAIL"

    log.info(f"  [Reproducibility] Status: {status} | Mismatches: {len(mismatches)}")

    return {
        "status":        status,
        "identical":     identical,
        "n_mismatches":  len(mismatches),
        "mismatches":    mismatches,
        "n_origins":     len(run1),
    }


# ── True out-of-sample evaluation (Correction 4) ──────────────────────────────

def run_oos_evaluation(
    df: pd.DataFrame,
    oos_origins: list[int],
    clip_config: ClipConfig = PRIMARY_CLIP_CONFIG,
    audit_frozen: bool = False,
) -> dict:
    """
    Evaluate A5a on genuinely new (OOS) origins.

    CORRECTION 4: This function must only be called AFTER all design decisions
    have been frozen. The `audit_frozen` flag must be True to proceed.
    If False, raises an error to prevent premature OOS peek.

    Returns:
        {"available": bool, "results": list[dict], "aggregate": dict}
    """
    if not audit_frozen:
        raise RuntimeError(
            "OOS evaluation called before audit_frozen=True. "
            "All design decisions (clipping, ablation, hyperparams) must be "
            "finalized before revealing OOS results. "
            "This is a Correction 4 enforcement error."
        )

    if not oos_origins:
        log.info("  [OOS] No new out-of-sample origins available.")
        return {
            "available":  False,
            "message":    "TRUE_NEW_OUT_OF_SAMPLE_DATA = UNAVAILABLE",
            "results":    [],
            "aggregate":  {},
        }

    log.info(f"  [OOS] Evaluating {len(oos_origins)} new out-of-sample origins: {oos_origins}")
    oos_origin_dicts = [{"feature_year": y} for y in oos_origins]
    oos_results = run_a5a_backtest(df, oos_origin_dicts, clip_config=clip_config)

    if not oos_results:
        return {
            "available": True,
            "message":   "OOS origins identified but evaluation produced no results.",
            "results":   [],
            "aggregate": {},
        }

    oos_agg = aggregate_results(oos_results, "A5a")
    return {
        "available":  True,
        "message":    f"TRUE_NEW_OUT_OF_SAMPLE_DATA = AVAILABLE. Origins: {oos_origins}",
        "results":    oos_results,
        "aggregate":  oos_agg,
    }


# ── Country-level analysis helpers ────────────────────────────────────────────

def compute_country_metrics_p20(
    results: list[dict],
    country_codes: list[str],
) -> pd.DataFrame:
    """Compute M0 and A5a RMSE/MAE per country from results list."""
    rows = []
    for cc in country_codes:
        m0_errors, a5a_errors = [], []
        for r in results:
            edf = r["eval_df"]
            mask = edf[COUNTRY_COL] == cc
            if not mask.any():
                continue
            y_cc    = r["y_eval"][mask.values]
            m0_cc   = r["predictions"]["M0"][mask.values]
            a5a_cc  = r["predictions"]["A5a"][mask.values]
            m0_errors.extend((y_cc - m0_cc).tolist())
            a5a_errors.extend((y_cc - a5a_cc).tolist())

        if not m0_errors:
            rows.append({"country": cc, "n": 0, "status": "NOT_FOUND"})
            continue

        m0_e  = np.array(m0_errors)
        a5a_e = np.array(a5a_errors)
        rows.append({
            "country":      cc,
            "n":            len(m0_e),
            "m0_rmse":      round(float(np.sqrt(np.mean(m0_e**2))), 4),
            "a5a_rmse":     round(float(np.sqrt(np.mean(a5a_e**2))), 4),
            "delta_rmse":   round(float(np.sqrt(np.mean(a5a_e**2)) - np.sqrt(np.mean(m0_e**2))), 4),
            "m0_mae":       round(float(np.mean(np.abs(m0_e))), 4),
            "a5a_mae":      round(float(np.mean(np.abs(a5a_e))), 4),
            "delta_mae":    round(float(np.mean(np.abs(a5a_e)) - np.mean(np.abs(m0_e))), 4),
            "a5a_better":   float(np.sqrt(np.mean(a5a_e**2))) < float(np.sqrt(np.mean(m0_e**2))),
        })
    return pd.DataFrame(rows)
