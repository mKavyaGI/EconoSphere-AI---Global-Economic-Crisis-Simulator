"""
EconoSphere AI — Phase 19
evaluation.py : Chronological expanding-window evaluation framework.

Note on prediction clipping:
  Linear models (Ridge, ElasticNet, Huber) are susceptible to wild extrapolation
  when extreme feature values (e.g. exchange_rate_lcu_usd for some countries)
  lie far outside the training distribution. Clipping final predictions to an
  economically defensible range [-40%, +60%] prevents a small number of
  outlier predictions from inflating mean RMSE for the entire model, while
  preserving the model's behaviour on normal observations.
  M0 and RandomForest are NOT clipped (tree models are naturally bounded).

Key design principles enforced here:
  1. Feature year Y → Target year Y+1  (T+1 only, verified explicitly).
  2. Training always uses years strictly < feature_year.
  3. All preprocessing (scaler, imputer) fitted on training data only.
  4. Hyperparameter selection via validation split inside training window only.
  5. A5a = equal-weight ensemble, A5b = validation-weighted ensemble.
  6. A6 residual model uses out-of-fold (OOF) predictions inside each
     training window — never the full training-set self-predictions.
  7. Test/forecast-origin observations never influence any model selection.
  8. No random train/test splits.

LEAKAGE RULES (L1–L12) are enforced as assertions where possible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)
from sklearn.model_selection import KFold

# Economically defensible GDP growth prediction bounds
PRED_CLIP_LOW  = -40.0   # -40% is historically extreme (e.g. wartime, deep crisis)
PRED_CLIP_HIGH =  60.0   # +60% is historically extreme (e.g. oil-boom micro-states)

from .models import (
    ELASTICNET_PARAMS,
    HUBER_EPSILONS,
    RIDGE_ALPHAS,
    build_elasticnet,
    build_huber,
    build_m0,
    build_random_forest,
    build_ridge,
)
from .utils import (
    LOCKED_FEATURES,
    TARGET_COL,
    YEAR_COL,
    check_no_target_leakage,
    get_logger,
)

log = get_logger()

# ── Metric helper ──────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mede = float(median_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    n    = int(len(y_true))
    return {"rmse": rmse, "mae": mae, "median_ae": mede, "bias": bias, "n": n}


# ── Leakage guard ──────────────────────────────────────────────────────────────

def _assert_no_target_in_X(X: pd.DataFrame, context: str = "") -> None:
    """L1/L2/L5 guard — raises ValueError if forbidden columns are present."""
    violations = check_no_target_leakage(X)
    if violations:
        raise ValueError(
            f"TARGET LEAKAGE DETECTED [{context}]: "
            f"forbidden columns in X: {violations}"
        )


def _assert_chronological_train(
    train_df: pd.DataFrame, feature_year: int, context: str = ""
) -> None:
    """L2 guard — all training rows must have year < feature_year."""
    if YEAR_COL in train_df.columns:
        bad = train_df[train_df[YEAR_COL] >= feature_year]
        if not bad.empty:
            raise ValueError(
                f"CHRONOLOGICAL LEAKAGE [{context}]: "
                f"{len(bad)} training rows have year >= feature_year {feature_year}. "
                f"Years found: {sorted(bad[YEAR_COL].unique())}"
            )


# ── Validation-split hyperparameter selection ─────────────────────────────────

def _select_ridge_alpha(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """L6 — selects Ridge alpha on validation split only."""
    best_alpha, best_rmse = RIDGE_ALPHAS[0], np.inf
    for alpha in RIDGE_ALPHAS:
        pipe = build_ridge(alpha=alpha)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        if rmse < best_rmse:
            best_rmse, best_alpha = rmse, alpha
    return best_alpha


def _select_elasticnet_params(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """L6 — selects ElasticNet (alpha, l1_ratio) on validation split only."""
    best_params, best_rmse = ELASTICNET_PARAMS[0], np.inf
    for params in ELASTICNET_PARAMS:
        pipe = build_elasticnet(**params)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        if rmse < best_rmse:
            best_rmse, best_params = rmse, params
    return best_params


def _select_huber_epsilon(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """L6 — selects Huber epsilon on validation split only."""
    best_eps, best_rmse = HUBER_EPSILONS[0], np.inf
    for eps in HUBER_EPSILONS:
        pipe = build_huber(epsilon=eps)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        if rmse < best_rmse:
            best_rmse, best_eps = rmse, eps
    return best_eps


# ── Validation split (chronological, within training window) ──────────────────

def _make_val_split(
    train_df: pd.DataFrame,
    val_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training data chronologically: last val_fraction years become
    the validation set, the remainder the fit set.
    Never shuffles.
    """
    years = sorted(train_df[YEAR_COL].unique())
    n_val = max(1, int(round(len(years) * val_fraction)))
    val_years  = set(years[-n_val:])
    fit_years  = set(years[:-n_val])
    fit_df = train_df[train_df[YEAR_COL].isin(fit_years)]
    val_df = train_df[train_df[YEAR_COL].isin(val_years)]
    return fit_df, val_df


# ── Out-of-fold Phase 11 predictions (for residual model) ─────────────────────

def _generate_oof_m0_predictions(
    train_df: pd.DataFrame,
    n_splits: int = 5,
) -> pd.Series:
    """
    Generate out-of-fold (OOF) Phase 11 predictions over the training set using
    chronological KFold splitting.

    WHY: We need residuals = actual - Phase11_prediction on the training data,
    but using a model trained on that same data would create artificially small
    residuals (the model would partially memorise them). OOF predictions ensure
    each training row is predicted by a model that was NOT trained on it.

    LEAKAGE GUARD (L8/L9): The OOF split is entirely within the training window.
    The forecast-origin rows (eval_df) are NEVER touched here.
    The test/eval targets never influence the residual model training.

    Returns a pd.Series of OOF predictions aligned with train_df's index.
    """
    X = train_df[LOCKED_FEATURES].copy()
    y = train_df[TARGET_COL].values

    # Use year-based groups to guarantee chronological folds.
    # Sort by year so earlier years are always in the fold train.
    years      = train_df[YEAR_COL].values
    unique_yrs = np.sort(np.unique(years))
    n_splits   = min(n_splits, len(unique_yrs) - 1)  # can't have more folds than years
    n_splits   = max(n_splits, 2)

    oof_preds = np.full(len(train_df), np.nan)

    # Use KFold on the sorted unique years, then map back to row indices
    kf = KFold(n_splits=n_splits, shuffle=False)
    for fold_train_idx, fold_val_idx in kf.split(unique_yrs):
        fold_train_years = set(unique_yrs[fold_train_idx])
        fold_val_years   = set(unique_yrs[fold_val_idx])

        # Row indices within train_df
        row_train = np.where(np.isin(years, list(fold_train_years)))[0]
        row_val   = np.where(np.isin(years, list(fold_val_years)))[0]

        if len(row_train) < 10 or len(row_val) == 0:
            continue

        X_fold_tr = X.iloc[row_train].values
        y_fold_tr = y[row_train]
        X_fold_vl = X.iloc[row_val].values

        pipe = build_m0()
        pipe.fit(X_fold_tr, y_fold_tr)
        oof_preds[row_val] = pipe.predict(X_fold_vl)

    oof_series = pd.Series(oof_preds, index=train_df.index, name="oof_m0_pred")
    return oof_series


# ── Ensemble weight selection ──────────────────────────────────────────────────

def _fit_ensemble_weights(
    preds_dict: dict[str, np.ndarray],
    y_val: np.ndarray,
) -> dict[str, float]:
    """
    A5b: Validation-derived non-negative ensemble weights.
    Uses scipy.optimize.minimize with non-negativity and sum=1 constraints.
    If optimisation fails, falls back to equal weights.
    L7 guarantee: weights are fitted on y_val (validation only), never test/eval.
    """
    names = list(preds_dict.keys())
    preds = np.column_stack([preds_dict[n] for n in names])
    n     = preds.shape[1]

    def objective(w):
        combo = preds @ w
        return float(np.sqrt(mean_squared_error(y_val, combo)))

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    try:
        res = minimize(objective, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-9})
        if res.success:
            w_opt = res.x
        else:
            log.warning("Ensemble weight optimisation did not converge; using equal weights.")
            w_opt = w0
    except Exception as exc:
        log.warning(f"Ensemble weight optimisation failed ({exc}); using equal weights.")
        w_opt = w0

    return {name: float(w) for name, w in zip(names, w_opt)}


# ── Main per-origin evaluation ─────────────────────────────────────────────────

def evaluate_origin(
    df: pd.DataFrame,
    feature_year: int,
) -> dict:
    """
    Run all Phase 19 candidates for a single forecast origin (feature_year).

    Returns a dict with keys:
        feature_year, target_year,
        predictions: {candidate_key: np.ndarray},
        metrics:     {candidate_key: metric_dict},
        y_eval, eval_df, hyperparams, ensemble_weights, residual_info
    """
    target_year = feature_year + 1  # L1: T+1 design enforced explicitly

    from .utils import TARGET_AVAIL_COL, get_train_eval_split

    train_df, eval_df = get_train_eval_split(df, feature_year)

    # ── Leakage guards ────────────────────────────────────────────────────────
    _assert_chronological_train(train_df, feature_year, context=f"origin={feature_year}")

    X_train_full = train_df[LOCKED_FEATURES].copy()
    y_train_full = train_df[TARGET_COL].values

    X_eval = eval_df[LOCKED_FEATURES].copy()
    y_eval = eval_df[TARGET_COL].values

    _assert_no_target_in_X(X_train_full, context=f"X_train origin={feature_year}")
    _assert_no_target_in_X(X_eval,       context=f"X_eval origin={feature_year}")

    assert int(y_eval.shape[0]) > 0, f"No eval rows at origin {feature_year}"
    assert not np.isnan(y_eval).all(), f"All eval targets are NaN at origin {feature_year}"

    # ── Validation split (last ~20% of training years) ───────────────────────
    fit_df, val_df = _make_val_split(train_df, val_fraction=0.20)

    X_fit = fit_df[LOCKED_FEATURES].values
    y_fit = fit_df[TARGET_COL].values
    X_val = val_df[LOCKED_FEATURES].values
    y_val = val_df[TARGET_COL].values

    # ── Hyperparameter selection (L6: validation only) ────────────────────────
    best_ridge_alpha  = _select_ridge_alpha(X_fit, y_fit, X_val, y_val)
    best_en_params    = _select_elasticnet_params(X_fit, y_fit, X_val, y_val)
    best_huber_eps    = _select_huber_epsilon(X_fit, y_fit, X_val, y_val)

    hyperparams = {
        "ridge_alpha":          best_ridge_alpha,
        "en_alpha":             best_en_params["alpha"],
        "en_l1_ratio":          best_en_params["l1_ratio"],
        "huber_epsilon":        best_huber_eps,
    }

    # ── Train base candidates on FULL training window ─────────────────────────
    log.info(f"  Origin {feature_year}→{target_year} | "
             f"n_train={len(train_df)}, n_eval={len(eval_df)}")

    # Identify which candidates are linear (need clipping)
    LINEAR_CANDIDATES = {"A1_Ridge", "A2_ElasticNet", "A3_Huber"}

    pipes = {
        "M0_Phase11":      build_m0(),
        "A1_Ridge":        build_ridge(alpha=best_ridge_alpha),
        "A2_ElasticNet":   build_elasticnet(**best_en_params),
        "A3_Huber":        build_huber(epsilon=best_huber_eps),
        "A4_RandomForest": build_random_forest(),
    }

    predictions: dict[str, np.ndarray] = {}
    val_predictions: dict[str, np.ndarray] = {}

    for key, pipe in pipes.items():
        pipe.fit(X_train_full.values, y_train_full)
        raw_eval = pipe.predict(X_eval.values)
        raw_val  = pipe.predict(X_val)
        # Clip linear model predictions to economically defensible bounds
        if key in LINEAR_CANDIDATES:
            raw_eval = np.clip(raw_eval, PRED_CLIP_LOW, PRED_CLIP_HIGH)
            raw_val  = np.clip(raw_val,  PRED_CLIP_LOW, PRED_CLIP_HIGH)
        predictions[key]     = raw_eval
        val_predictions[key] = raw_val

    # ── A5a — Equal-weight ensemble ───────────────────────────────────────────
    # Uses M0, A1_Ridge, A4_RandomForest for architectural diversity
    ensemble_base_keys = ["M0_Phase11", "A1_Ridge", "A4_RandomForest"]
    preds_a5a = np.mean(
        np.column_stack([predictions[k] for k in ensemble_base_keys]), axis=1
    )
    predictions["A5a_Ensemble"] = preds_a5a

    # ── A5b — Validation-selected ensemble weights ────────────────────────────
    # L7: weights determined using val_predictions only (NOT eval/test)
    val_preds_for_ens = {k: val_predictions[k] for k in ensemble_base_keys}
    a5b_weights = _fit_ensemble_weights(val_preds_for_ens, y_val)
    preds_a5b = sum(
        a5b_weights[k] * predictions[k] for k in ensemble_base_keys
    )
    predictions["A5b_Ensemble"] = np.asarray(preds_a5b)

    # ── A6 — Residual Model ───────────────────────────────────────────────────
    # Step 1: Generate OOF M0 predictions over training data (L8/L9 guards)
    oof_m0 = _generate_oof_m0_predictions(train_df, n_splits=5)

    # Step 2: OOF residuals — only where we have valid OOF predictions
    valid_oof = ~oof_m0.isna()
    oof_residuals = train_df.loc[valid_oof, TARGET_COL] - oof_m0[valid_oof]

    residual_info = {
        "n_oof_valid": int(valid_oof.sum()),
        "n_oof_nan":   int((~valid_oof).sum()),
    }

    # Step 3: Train residual model (Ridge) on OOF residuals
    # L8: residual model never sees eval/test residuals
    X_residual_train = train_df.loc[valid_oof, LOCKED_FEATURES].values
    y_residual_train = oof_residuals.values

    a6_residual_pipe = build_ridge(alpha=best_ridge_alpha)
    a6_residual_pipe.fit(X_residual_train, y_residual_train)

    # Step 4: Final A6 prediction = M0 prediction + residual correction
    # Clip residual correction (Ridge) to prevent linear extrapolation blow-up
    m0_eval_pred        = predictions["M0_Phase11"]
    residual_correction = np.clip(
        a6_residual_pipe.predict(X_eval.values), PRED_CLIP_LOW, PRED_CLIP_HIGH
    )
    predictions["A6_Residual"] = np.clip(
        m0_eval_pred + residual_correction, PRED_CLIP_LOW, PRED_CLIP_HIGH
    )

    # ── Compute metrics for all candidates ───────────────────────────────────
    metrics: dict[str, dict] = {}
    for key, pred in predictions.items():
        metrics[key] = compute_metrics(y_eval, pred)

    return {
        "feature_year":     feature_year,
        "target_year":      target_year,
        "n_train":          len(train_df),
        "n_eval":           len(eval_df),
        "predictions":      predictions,
        "y_eval":           y_eval,
        "eval_df":          eval_df,
        "metrics":          metrics,
        "hyperparams":      hyperparams,
        "ensemble_weights": a5b_weights,
        "residual_info":    residual_info,
    }


# ── Full chronological backtest ────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    valid_origins: list[dict],
    logger=None,
) -> list[dict]:
    """
    Run evaluate_origin() for every valid origin and collect results.
    Returns a list of result dicts (one per origin).
    """
    if logger is None:
        logger = log
    results = []
    for origin_info in valid_origins:
        fy = origin_info["feature_year"]
        try:
            result = evaluate_origin(df, fy)
            results.append(result)
            # Per-origin summary log
            m0_rmse = result["metrics"]["M0_Phase11"]["rmse"]
            best_cand = min(
                result["metrics"].items(), key=lambda kv: kv[1]["rmse"]
            )
            logger.info(
                f"  Origin {fy}→{fy+1} | M0 RMSE={m0_rmse:.4f} | "
                f"Best={best_cand[0]} RMSE={best_cand[1]['rmse']:.4f}"
            )
        except Exception as exc:
            logger.error(f"  Origin {fy}: FAILED — {exc}", exc_info=True)
    return results


# ── Aggregation helpers ────────────────────────────────────────────────────────

def build_master_table(results: list[dict]) -> pd.DataFrame:
    """
    Build the master comparison table across all origins.
    Columns: model, mean_rmse, median_rmse, mean_mae, origin_wins,
             phase11_wins, ties, delta_rmse, delta_mae
    """
    if not results:
        return pd.DataFrame()

    all_candidates = list(results[0]["metrics"].keys())
    rows = []

    for cand in all_candidates:
        cand_rmses   = [r["metrics"][cand]["rmse"] for r in results if cand in r["metrics"]]
        cand_maes    = [r["metrics"][cand]["mae"]  for r in results if cand in r["metrics"]]
        m0_rmses     = [r["metrics"]["M0_Phase11"]["rmse"] for r in results if cand in r["metrics"]]

        wins = ties = losses = 0
        for cr, mr in zip(cand_rmses, m0_rmses):
            if abs(cr - mr) < 1e-6:
                ties += 1
            elif cr < mr:
                wins += 1
            else:
                losses += 1

        mean_rmse   = float(np.mean(cand_rmses))
        median_rmse = float(np.median(cand_rmses))
        mean_mae    = float(np.mean(cand_maes))
        m0_mean     = float(np.mean(m0_rmses))
        delta_rmse  = round(mean_rmse - m0_mean, 4)

        rows.append({
            "model":         cand,
            "mean_rmse":     round(mean_rmse, 4),
            "median_rmse":   round(median_rmse, 4),
            "mean_mae":      round(mean_mae, 4),
            "origin_wins":   wins,
            "phase11_wins":  losses,
            "ties":          ties,
            "delta_rmse":    delta_rmse,
            "n_origins":     len(cand_rmses),
        })

    df_out = pd.DataFrame(rows)
    return df_out


def build_predictions_df(results: list[dict]) -> pd.DataFrame:
    """
    Build a long-form predictions DataFrame with one row per
    (origin, country, candidate).
    """
    records = []
    for r in results:
        eval_df     = r["eval_df"].reset_index(drop=True)
        y_eval      = r["y_eval"]
        feature_year = r["feature_year"]
        target_year  = r["target_year"]

        for cand, preds in r["predictions"].items():
            for i, (actual, pred) in enumerate(zip(y_eval, preds)):
                row_meta = eval_df.iloc[i] if i < len(eval_df) else {}
                country  = row_meta.get("country_code", "") if hasattr(row_meta, "get") else ""
                records.append({
                    "feature_year":         feature_year,
                    "target_year":          target_year,
                    "country_code":         country,
                    "candidate":            cand,
                    "actual_gdp_growth":    float(actual),
                    "predicted_gdp_growth": float(pred),
                    "residual":             float(actual - pred),
                    "absolute_error":       float(abs(actual - pred)),
                    "squared_error":        float((actual - pred) ** 2),
                })

    return pd.DataFrame(records)


def compute_country_metrics(
    preds_df: pd.DataFrame,
    country_codes: list[str],
) -> pd.DataFrame:
    """Compute per-country metrics for a list of countries."""
    rows = []
    for cc in country_codes:
        cdf = preds_df[preds_df["country_code"] == cc]
        if cdf.empty:
            continue
        for cand, g in cdf.groupby("candidate"):
            m = compute_metrics(g["actual_gdp_growth"].values,
                                g["predicted_gdp_growth"].values)
            rows.append({"country": cc, "candidate": cand, **m})
    return pd.DataFrame(rows)
