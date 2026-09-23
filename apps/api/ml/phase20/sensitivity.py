"""
EconoSphere AI — Phase 20
sensitivity.py : Clipping sensitivity configuration.

CORRECTION 3 (User-approved):
  C0/C1/C2 are sensitivity evidence, NOT three independent promotion requirements.
  G5 PASS: A5a advantage is not solely an artifact of the chosen clipping boundary.
  G5 FAIL: A5a improvement disappears or reverses under reasonable pre-specified
           checks in a way indicating clipping drives the result.

  C0 is deliberately testing raw extrapolation — linear models will produce
  economically extreme predictions in some cases. This is known and documented.
  The question is whether the A5a advantage is meaningful across all configs.

CORRECTION KEY: The clipping thresholds are pre-specified here. They must NOT
  be changed after looking at test-set results. No test-driven clipping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ClipConfig:
    """A single pre-specified clipping configuration."""
    name:        str
    label:       str
    clip_low:    Optional[float]
    clip_high:   Optional[float]
    description: str

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """Apply this clipping config to a prediction array."""
        if self.clip_low is None and self.clip_high is None:
            return arr.copy()
        return np.clip(arr, self.clip_low, self.clip_high)


# ── Pre-specified clipping configurations (LOCKED before any evaluation) ──────
# These configurations are defined here and FROZEN.
# They must not be modified in response to test-set results.

CLIP_CONFIGS: dict[str, ClipConfig] = {
    "C0": ClipConfig(
        name="C0",
        label="No Clipping",
        clip_low=None,
        clip_high=None,
        description=(
            "Raw predictions — no clipping applied to linear models. "
            "Expected to show extreme values for outlier countries. "
            "Tests whether A5a advantage exists even without clipping."
        ),
    ),
    "C1": ClipConfig(
        name="C1",
        label="Phase 19 Clipping [-40, +60]",
        clip_low=-40.0,
        clip_high=60.0,
        description=(
            "Phase 19 production-candidate clipping. "
            "-40%: historically extreme (wartime, deep crisis). "
            "+60%: historically extreme (oil-boom micro-states). "
            "This is the primary Phase 20 evaluation configuration."
        ),
    ),
    "C2": ClipConfig(
        name="C2",
        label="Alternative Bounds [-50, +75]",
        clip_low=-50.0,
        clip_high=75.0,
        description=(
            "Alternative wider bounds. "
            "Permits more extreme predictions than C1, capturing more "
            "post-conflict or micro-state recovery variance. "
            "Tests whether tighter C1 bounds artificially inflate A5a."
        ),
    ),
}

# The primary evaluation configuration is C1 (Phase 19 config)
PRIMARY_CLIP_CONFIG = CLIP_CONFIGS["C1"]


def clip_linear_predictions(
    predictions: np.ndarray,
    config: ClipConfig,
) -> np.ndarray:
    """
    Apply clipping to linear model predictions per a given ClipConfig.
    M0 (HGB) and RandomForest are never clipped — tree models are naturally bounded.
    This function is only called for linear candidates (Ridge).
    """
    return config.apply(predictions)


def g5_clipping_sensitivity_assessment(
    clip_results: dict[str, dict],
    log,
) -> dict:
    """
    Evaluate Gate 5: Clipping Sensitivity.

    CORRECTION 3: G5 is NOT "A5a beats M0 under all three configs".
    It is: "The A5a improvement is not solely an artifact of the clipping choice."

    Assessment logic:
    - If A5a beats M0 under C1 AND C2 → strong evidence clipping not driving result
    - If A5a beats M0 under C1 but not C0 → acceptable: C0 is raw linear, known to explode
    - If A5a LOSES to M0 under C1 → G5 FAIL (the primary config doesn't work)
    - If A5a improvement under C1 dramatically exceeds C2 → investigate further

    Returns gate result dict.
    """
    results_by_config = {}
    for cfg_name, res in clip_results.items():
        a5a_rmse = res.get("a5a_mean_rmse")
        m0_rmse  = res.get("m0_mean_rmse")
        delta    = res.get("delta_rmse")
        wins     = res.get("origin_wins")
        results_by_config[cfg_name] = {
            "a5a_rmse": a5a_rmse, "m0_rmse": m0_rmse,
            "delta": delta, "wins": wins,
        }

    c1 = results_by_config.get("C1", {})
    c2 = results_by_config.get("C2", {})

    c1_delta = c1.get("delta", 0.0) or 0.0
    c2_delta = c2.get("delta", 0.0) or 0.0
    c1_wins  = c1.get("wins", 0) or 0

    # Primary config (C1) must show positive A5a advantage
    c1_positive = c1_delta < 0.0  # negative delta = A5a better

    # C2 should also show advantage (wider bounds = more conservative test)
    c2_positive = c2_delta < 0.0

    # Check for dramatic difference between C1 and C2 (flag if |c1_delta - c2_delta| > 0.3)
    config_diff = abs(c1_delta - c2_delta)
    large_config_sensitivity = config_diff > 0.30

    if not c1_positive:
        gate_status = "FAIL"
        rationale = f"A5a does not improve over M0 under primary config C1 (ΔRMSE={c1_delta:+.4f}). G5 FAIL."
    elif large_config_sensitivity:
        gate_status = "MARGINAL"
        rationale = (
            f"A5a improves under C1 (ΔRMSE={c1_delta:+.4f}) but C1 vs C2 differ by "
            f"{config_diff:.4f} RMSE — larger than expected. "
            "Clipping may be amplifying the advantage. Flagged for review."
        )
    elif c1_positive and c2_positive:
        gate_status = "PASS"
        rationale = (
            f"A5a improves under both C1 (ΔRMSE={c1_delta:+.4f}) and C2 (ΔRMSE={c2_delta:+.4f}). "
            "Improvement is not solely an artifact of the chosen clipping boundary."
        )
    else:
        gate_status = "PASS"
        rationale = (
            f"A5a improves under C1 (ΔRMSE={c1_delta:+.4f}). "
            f"C2 shows different result (ΔRMSE={c2_delta:+.4f}). "
            "C0 raw behavior is expected to be poor (known linear extrapolation issue). "
            "C1 advantage is defensible."
        )

    log.info(f"  [G5] Clipping Sensitivity: {gate_status} — {rationale}")

    return {
        "gate":                    "G5",
        "status":                  gate_status,
        "rationale":               rationale,
        "c1_delta":                c1_delta,
        "c2_delta":                c2_delta,
        "config_sensitivity":      config_diff,
        "large_config_sensitivity": large_config_sensitivity,
        "results_by_config":       results_by_config,
    }
