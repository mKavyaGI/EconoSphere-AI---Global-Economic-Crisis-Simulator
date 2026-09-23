"""
EconoSphere AI — Phase 21
run_validation.py : Main orchestrator for Phase 21 Blind True OOS Validation.

Execution order (§31):
  Step 1  — Inspect Phase 19/20 (done in research phase)
  Step 2  — Identify frozen A5a artifacts
  Step 3  — Hash production artifacts (before)
  Step 4  — Create pre-OOS lock [Stage A]
  Step 5  — Discover genuinely new OOS data [Stage B]
  Step 6  — Freeze OOS origins [Stage C] + create inconclusive report if unavailable
  Step 7  — Blind inference + reveal [Stages D+E] (only if OOS data exists)
  Step 8  — Run all promotion gates (G1-G12)
  Step 9  — Hash production artifacts (after)
  Step 10 — (tests run externally via pytest)
  Step 11 — (full regression run externally)
  Step 12 — Generate consolidated governance report
  Step 13 — DO NOT promote automatically

GOVERNANCE:
  - No model is modified, retrained, or replaced.
  - No production artifact is touched.
  - No post-hoc tuning occurs after OOS labels are read.
  - Governance state is determined programmatically.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .evaluation import (
    check_leakage_rules,
    compute_gate_results,
    determine_governance_state,
    run_artifact_test,
    run_oos_evaluation,
    run_production_immutability_test,
    run_reproducibility_check,
)
from .lock import create_pre_evaluation_lock, freeze_oos_origins
from .reports import generate_all_reports, print_terminal_summary
from .utils import (
    PHASE21_DOCS_DIR,
    discover_oos_origins,
    get_logger,
    load_dataset,
    record_hashes_p21,
    verify_hash_immutability,
)

log = get_logger()


def main() -> str:
    """
    Run the full Phase 21 governance protocol.
    Returns the final governance state string.
    """
    log.info("=" * 60)
    log.info("PHASE 21 — BLIND TRUE OOS VALIDATION OF FROZEN A5a")
    log.info("=" * 60)
    log.info("GOVERNANCE: Validation-only phase. No model development.")
    log.info("GOVERNANCE: No production artifact will be modified.")

    PHASE21_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 2: Identify frozen A5a artifacts ─────────────────────────────────
    log.info("\n=== Step 2: Identifying frozen A5a experimental artifacts ===")
    from .utils import A5A_M0_PATH, A5A_RIDGE_PATH, A5A_RF_PATH, A5A_META_PATH
    for path in [A5A_M0_PATH, A5A_RIDGE_PATH, A5A_RF_PATH, A5A_META_PATH]:
        exists = path.exists()
        log.info(f"  {path.name}: {'FOUND' if exists else 'MISSING'}")
        if not exists and path.suffix == ".joblib":
            log.warning(f"  WARNING: Experimental artifact missing: {path}")

    # ── Step 3: Hash production artifacts (before) ────────────────────────────
    log.info("\n=== Step 3: Hashing production artifacts (before) ===")
    before_hashes = record_hashes_p21("before")
    for name, h in before_hashes.items():
        log.info(f"  {name}: md5={h['md5'][:16]}...")

    # ── Step 4: Create pre-evaluation lock [Stage A] ──────────────────────────
    log.info("\n=== Step 4 / Stage A: Creating pre-evaluation lock ===")
    lock = create_pre_evaluation_lock()
    log.info("  Lock created. OOS labels have NOT been read.")

    # ── Load dataset (read-only) ───────────────────────────────────────────────
    log.info("\n=== Loading dataset (read-only) ===")
    df = load_dataset()
    log.info(f"  Dataset loaded: {len(df)} rows, max year={df['year'].max()}")

    # ── Step 5: OOS Discovery [Stage B] ───────────────────────────────────────
    # CORRECTION 3: Discovery reads ONLY availability metadata, NOT target values.
    # Discovery cannot modify the lock.
    log.info("\n=== Step 5 / Stage B: OOS Discovery (reads availability metadata only) ===")
    oos_discovery = discover_oos_origins(df, log)
    log.info(f"  OOS available: {oos_discovery['available']}")
    log.info(f"  OOS origins:   {oos_discovery['oos_origins']}")

    # ── Step 6: Freeze OOS origins [Stage C] ─────────────────────────────────
    log.info("\n=== Step 6 / Stage C: Freezing OOS origins ===")
    frozen_origins = freeze_oos_origins(oos_discovery)

    # ── Step 7: Blind inference + reveal [Stages D+E] ─────────────────────────
    log.info("\n=== Step 7 / Stages D+E: Blind OOS Evaluation ===")
    oos_eval = run_oos_evaluation(df, oos_discovery["oos_origins"])

    if not oos_discovery["available"]:
        log.info("  OOS data unavailable — blind inference skipped.")
        log.info("  CORRECTION 4: CSV outputs will be annotated NO_OOS_DATA, not empty.")

    # ── Leakage check ─────────────────────────────────────────────────────────
    log.info("\n=== Leakage check (L1-L12) ===")
    leakage = check_leakage_rules(df)
    for rule, status in leakage.items():
        log.info(f"  {rule}: {status}")

    # ── Artifact test ─────────────────────────────────────────────────────────
    log.info("\n=== Operational / Artifact test ===")
    artifact_test = run_artifact_test(df)
    log.info(f"  Artifact test: {artifact_test['status']}")

    # ── Reproducibility ───────────────────────────────────────────────────────
    log.info("\n=== Reproducibility check ===")
    reproducibility = run_reproducibility_check(df, oos_discovery["oos_origins"])
    log.info(f"  Reproducibility: {reproducibility['status']}")

    # ── Step 9: Hash production artifacts (after) ─────────────────────────────
    log.info("\n=== Step 9: Hashing production artifacts (after) ===")
    after_hashes = record_hashes_p21("after")
    immutability = run_production_immutability_test(before_hashes, after_hashes)
    log.info(f"  Production immutability: {immutability['status']}")

    if not immutability["PRODUCTION_ARTIFACTS_UNCHANGED"]:
        log.error("  GOVERNANCE FAILURE: Production artifact hash changed!")

    # ── Step 8: Run all promotion gates (G1-G12) ──────────────────────────────
    log.info("\n=== Step 8: Evaluating promotion gates (G1-G12) ===")
    gates = compute_gate_results(
        immutability=immutability,
        oos_discovery=oos_discovery,
        oos_eval=oos_eval,
        artifact_test=artifact_test,
        reproducibility=reproducibility,
        leakage_checks=leakage,
    )
    for g, s in gates.items():
        log.info(f"  {g}: {s}")

    # ── Determine governance state programmatically ────────────────────────────
    governance_state = determine_governance_state(gates, oos_eval, immutability)
    log.info(f"\n  GOVERNANCE STATE: {governance_state}")
    log.info("  PRODUCTION STATUS: FROZEN_PRODUCTION_RETAINED")
    log.info("  NO AUTOMATIC PROMOTION.")

    # ── Step 12: Generate all reports ─────────────────────────────────────────
    log.info("\n=== Step 12: Generating Phase 21 reports ===")
    generate_all_reports(
        audit_data={},
        lock=lock,
        frozen_origins=frozen_origins,
        oos_discovery=oos_discovery,
        oos_eval=oos_eval,
        gates=gates,
        governance_state=governance_state,
        immutability=immutability,
        reproducibility=reproducibility,
        artifact_test=artifact_test,
        leakage=leakage,
        before_hashes=before_hashes,
        after_hashes=after_hashes,
    )

    # ── Final terminal summary (§35) ──────────────────────────────────────────
    print_terminal_summary(
        governance_state=governance_state,
        oos_eval=oos_eval,
        immutability=immutability,
        reproducibility=reproducibility,
        test_result="RUN SEPARATELY: pytest apps/api/tests/test_phase21_oos_validation.py",
        regression_result="RUN SEPARATELY: pytest apps/api/tests/ -q",
    )

    # ── Step 13: Final governance reminder ────────────────────────────────────
    log.info("\n=== Step 13: Governance reminder ===")
    log.info("  Phase 11 production model is UNCHANGED.")
    log.info("  No promotion has occurred.")
    log.info("  No experimental artifact has been renamed to a production filename.")
    log.info(f"  Phase 21 complete. Decision: {governance_state}")

    return governance_state


if __name__ == "__main__":
    main()
