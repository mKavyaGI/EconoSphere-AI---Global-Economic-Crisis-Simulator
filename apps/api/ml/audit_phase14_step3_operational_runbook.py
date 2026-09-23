"""
EconoSphere AI — Phase 14 Step 3: Operational Runbook Audit
===================================================================
Non-destructive operational audit script to verify that the repository
configuration matches the documented assumptions in Phase 14 Step 3.

Checks:
1. Documentation files exist
2. Protected artifacts exist
3. Artifact hashes match frozen baseline
4. Dockerfile exists
5. .dockerignore exists and excludes .venv
6. docker-compose.yml exists
7. Compose configuration contains read-only mounts
8. Production safety variables configured
9. No wildcard CORS allowed
10. Step 2 audit exists
"""

import os
import sys
import json
import csv
import time
import hashlib
from pathlib import Path
import yaml

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = PROJECT_ROOT / "docs" / "phase14"
DOCKER_DIR = PROJECT_ROOT / "infrastructure" / "docker"
COMPOSE_FILE = DOCKER_DIR / "docker-compose.yml"
API_DIR = PROJECT_ROOT / "apps" / "api"
DOCKERFILE = API_DIR / "Dockerfile"
DOCKERIGNORE_FILE = API_DIR / ".dockerignore"

DOCS_TO_CHECK = [
    DOCS_DIR / "step3_production_deployment_runbook.md",
    DOCS_DIR / "step3_production_quickstart.md",
    DOCS_DIR / "step3_production_deployment_checklist.md"
]

ARTIFACTS = {
    "model": PROJECT_ROOT / "models" / "phase11" / "best_t1_gdp_growth_model.joblib",
    "manifest": PROJECT_ROOT / "models" / "phase11" / "phase11_production_release_manifest.json",
    "dataset": PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
}

# Approved Phase 11 Baselines
EXPECTED_MD5 = {
    "model": "9c539735897eaf6e72e5f54f047390d7",
    "manifest": "9c673e2d5bcd12adc171a55bd86ca34e",
    "dataset": "8ac7e0b2bf09fbe89289f82d0c7cf25e"
}

results = {
    "Runbook Documentation Exists": "FAIL",
    "Quickstart Documentation Exists": "FAIL",
    "Checklist Documentation Exists": "FAIL",
    "Protected Model Artifact Exists": "FAIL",
    "Protected Manifest Artifact Exists": "FAIL",
    "Protected Dataset Artifact Exists": "FAIL",
    "Model Hash Verified (Immutable)": "FAIL",
    "Manifest Hash Verified (Immutable)": "FAIL",
    "Dataset Hash Verified (Immutable)": "FAIL",
    "Dockerfile Exists": "FAIL",
    ".dockerignore Exists & Excludes .venv": "FAIL",
    "docker-compose.yml Exists": "FAIL",
    "Compose Model Mount Read-Only": "FAIL",
    "Compose Data Mount Read-Only": "FAIL",
    "Compose Production ENV Configured": "FAIL",
    "Compose Safety Mode Configured": "FAIL",
    "Compose No Wildcard CORS": "FAIL",
    "Phase 14 Step 2 Audit Exists": "FAIL"
}

def get_hash(filepath: Path, algo="md5") -> str:
    hash_func = hashlib.md5() if algo == "md5" else hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def run_audit():
    print("=" * 75)
    print("ECONOSPHERE AI — PHASE 14 STEP 3 OPERATIONAL RUNBOOK AUDIT")
    print("=" * 75)
    
    # 1. Documentation Checks
    print("[1/6] Verifying Documentation Presence...")
    if DOCS_TO_CHECK[0].exists(): results["Runbook Documentation Exists"] = "PASS"
    if DOCS_TO_CHECK[1].exists(): results["Quickstart Documentation Exists"] = "PASS"
    if DOCS_TO_CHECK[2].exists(): results["Checklist Documentation Exists"] = "PASS"

    # 2. Artifact Presence & Integrity
    print("[2/6] Verifying Protected Artifact Immutability...")
    if ARTIFACTS["model"].exists(): results["Protected Model Artifact Exists"] = "PASS"
    if ARTIFACTS["manifest"].exists(): results["Protected Manifest Artifact Exists"] = "PASS"
    if ARTIFACTS["dataset"].exists(): results["Protected Dataset Artifact Exists"] = "PASS"

    pre_hashes = {}
    for key, path in ARTIFACTS.items():
        if path.exists():
            h = get_hash(path, "md5")
            pre_hashes[key] = h
            if h == EXPECTED_MD5[key]:
                results[f"{key.capitalize()} Hash Verified (Immutable)"] = "PASS"
            else:
                print(f" -> ERROR: Hash mismatch for {key}. Expected {EXPECTED_MD5[key]}, got {h}")

    # 3. Docker Config Checks
    print("[3/6] Verifying Docker Configuration files...")
    if DOCKERFILE.exists(): results["Dockerfile Exists"] = "PASS"
    if DOCKERIGNORE_FILE.exists():
        content = DOCKERIGNORE_FILE.read_text()
        if ".venv" in content:
            results[".dockerignore Exists & Excludes .venv"] = "PASS"

    # 4. Compose Configuration Checks
    print("[4/6] Verifying Docker Compose Constraints...")
    if COMPOSE_FILE.exists():
        results["docker-compose.yml Exists"] = "PASS"
        try:
            with open(COMPOSE_FILE, "r") as f:
                config = yaml.safe_load(f)
                api_svc = config.get("services", {}).get("api", {})
                
                # Mounts
                vols = api_svc.get("volumes", [])
                for v in vols:
                    if isinstance(v, str):
                        if "/models:ro" in v: results["Compose Model Mount Read-Only"] = "PASS"
                        if "/data:ro" in v: results["Compose Data Mount Read-Only"] = "PASS"
                    elif isinstance(v, dict):
                        if "/models" in v.get("target", "") and v.get("read_only"): results["Compose Model Mount Read-Only"] = "PASS"
                        if "/data" in v.get("target", "") and v.get("read_only"): results["Compose Data Mount Read-Only"] = "PASS"
                
                # Envs
                envs = api_svc.get("environment", {})
                if envs.get("ENV") == "production": results["Compose Production ENV Configured"] = "PASS"
                if str(envs.get("PRODUCTION_SAFETY_MODE")).lower() == "true": results["Compose Safety Mode Configured"] = "PASS"
                
                origins = envs.get("ALLOWED_ORIGINS", "")
                if origins and "*" not in origins: results["Compose No Wildcard CORS"] = "PASS"
        except Exception as e:
            print(f" -> ERROR reading compose file: {e}")

    # 5. Historical Audit Check
    print("[5/6] Verifying Historical Step 2 Audit...")
    step2_audit = API_DIR / "ml" / "audit_phase14_step2_containerized_runtime.py"
    if step2_audit.exists(): results["Phase 14 Step 2 Audit Exists"] = "PASS"

    # 6. Post-Audit Immutability Verification
    print("[6/6] Verifying Post-Audit Artifact Immutability...")
    post_hashes = {}
    hash_failure = False
    for key, path in ARTIFACTS.items():
        if path.exists():
            post_hashes[key] = get_hash(path, "md5")
            if post_hashes[key] != pre_hashes.get(key):
                hash_failure = True
                print(f" -> CRITICAL ERROR: Hash mutated during audit for {key}!")
    
    if hash_failure:
        print("\n>>> CRITICAL FAILURE: PROTECTED ARTIFACTS MUTATED DURING AUDIT <<<")
        sys.exit(1)

    # Evaluation
    all_pass = all(v == "PASS" for v in results.values())
    
    print("\n" + "=" * 75)
    print("PHASE 14 STEP 3 — OPERATIONAL RUNBOOK AUDIT SUMMARY")
    print("=" * 75)
    for check_name, status in results.items():
        print(f" {check_name.ljust(50)}: {status}")
    print("=" * 75)

    # Save outputs
    out_dir = DOCS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = out_dir / "phase14_step3_operational_validation_metadata.json"
    meta = {
        "phase": "14",
        "step": "3",
        "purpose": "Operational Runbook and Documentation Verification",
        "production_release_id": "ECONOSPHERE-PHASE11-PROD-2026-08-21",
        "overall_status": "PASS" if all_pass else "FAIL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pre_audit_hashes": pre_hashes,
        "post_audit_hashes": post_hashes,
        "checks": results
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    csv_path = out_dir / "phase14_step3_operational_validation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Check", "Status"])
        for k, v in results.items():
            writer.writerow([k, v])

    print("\nFINAL DECISION:")
    if all_pass:
        print("\n>>> PHASE 14 STEP 3 — OPERATIONAL RUNBOOK VERIFIED <<<\n")
    else:
        print("\n>>> PHASE 14 STEP 3 — DOCUMENTATION CREATED, VERIFICATION FAILED <<<\n")

if __name__ == "__main__":
    run_audit()
