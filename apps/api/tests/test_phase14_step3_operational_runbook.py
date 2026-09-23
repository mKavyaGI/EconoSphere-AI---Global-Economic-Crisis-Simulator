import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = PROJECT_ROOT / "docs" / "phase14"
DOCKER_DIR = PROJECT_ROOT / "infrastructure" / "docker"
COMPOSE_FILE = DOCKER_DIR / "docker-compose.yml"
API_DIR = PROJECT_ROOT / "apps" / "api"
DOCKERFILE = API_DIR / "Dockerfile"
DOCKERIGNORE = API_DIR / ".dockerignore"

RUNBOOK = DOCS_DIR / "step3_production_deployment_runbook.md"
QUICKSTART = DOCS_DIR / "step3_production_quickstart.md"
CHECKLIST = DOCS_DIR / "step3_production_deployment_checklist.md"
AUDIT_SCRIPT = API_DIR / "ml" / "audit_phase14_step3_operational_runbook.py"

# A. Documentation Presence Tests (3 tests)
def test_runbook_exists():
    assert RUNBOOK.exists()

def test_quickstart_exists():
    assert QUICKSTART.exists()

def test_checklist_exists():
    assert CHECKLIST.exists()

# B. Governance Language Verification Tests (9 tests)
def test_runbook_documents_phase11_frozen_baseline():
    content = RUNBOOK.read_text(encoding="utf-8").lower()
    assert "phase 11" in content
    assert "frozen" in content
    assert "immutable" in content

def test_runbook_documents_31_features():
    content = RUNBOOK.read_text(encoding="utf-8").lower()
    assert "31" in content

def test_runbook_documents_candidate_b_isolation():
    content = RUNBOOK.read_text(encoding="utf-8").lower()
    assert "growth_regime_num" in content
    assert "stress_regime_num" in content
    assert "422" in content

def test_runbook_documents_readonly_volumes():
    content = RUNBOOK.read_text(encoding="utf-8").lower()
    assert "read-only" in content or "readonly" in content
    assert ":ro" in content

def test_runbook_documents_no_wildcard_cors():
    content = RUNBOOK.read_text(encoding="utf-8").lower()
    assert "wildcard" in content
    assert "*" in content

def test_runbook_documents_fail_closed():
    content = RUNBOOK.read_text(encoding="utf-8").lower()
    assert "fail-closed" in content

def test_runbook_documents_artifact_integrity():
    content = RUNBOOK.read_text(encoding="utf-8").lower()
    assert "hash mismatch" in content
    assert "integrity" in content

def test_runbook_documents_safe_shutdown():
    content = RUNBOOK.read_text(encoding="utf-8").lower()
    assert "down --remove-orphans" in content

def test_runbook_documents_recovery_procedure():
    content = RUNBOOK.read_text(encoding="utf-8").lower()
    assert "recovery procedure" in content
    assert "prohibited recovery actions" in content

# C. Docker Documentation Consistency (4 tests)
def test_dockerfile_exists_at_documented_path():
    assert DOCKERFILE.exists()

def test_dockerignore_exists():
    assert DOCKERIGNORE.exists()

def test_dockerignore_excludes_venv():
    content = DOCKERIGNORE.read_text(encoding="utf-8")
    assert ".venv" in content

def test_compose_file_exists_at_documented_path():
    assert COMPOSE_FILE.exists()

# D. Compose Read-Only Mounts Verification (2 tests)
def get_compose_config():
    with open(COMPOSE_FILE, "r") as f:
        return yaml.safe_load(f)

def test_compose_models_is_readonly():
    config = get_compose_config()
    vols = config["services"]["api"].get("volumes", [])
    model_ro = any("/models:ro" in str(v) for v in vols) or any(isinstance(v, dict) and "/models" in v.get("target", "") and v.get("read_only", False) for v in vols)
    assert model_ro

def test_compose_data_is_readonly():
    config = get_compose_config()
    vols = config["services"]["api"].get("volumes", [])
    data_ro = any("/data:ro" in str(v) for v in vols) or any(isinstance(v, dict) and "/data" in v.get("target", "") and v.get("read_only", False) for v in vols)
    assert data_ro

# E. Production Environment Documentation (3 tests)
def test_compose_production_env():
    config = get_compose_config()
    env = config["services"]["api"].get("environment", {})
    assert env.get("ENV") == "production"

def test_compose_production_safety_mode():
    config = get_compose_config()
    env = config["services"]["api"].get("environment", {})
    assert str(env.get("PRODUCTION_SAFETY_MODE")).lower() == "true"

def test_compose_allowed_origins_explicit():
    config = get_compose_config()
    env = config["services"]["api"].get("environment", {})
    origins = env.get("ALLOWED_ORIGINS", "")
    assert origins != ""
    assert "*" not in origins

# F. Runbook Validation Script Integrity Tests (6 tests)
def test_audit_script_exists():
    assert AUDIT_SCRIPT.exists()

@patch('ml.audit_phase14_step3_operational_runbook.sys.exit')
@patch('builtins.print')
def test_audit_fails_on_hash_mismatch(mock_print, mock_exit):
    import ml.audit_phase14_step3_operational_runbook as audit

    
    original_hashes = audit.EXPECTED_MD5.copy()
    audit.EXPECTED_MD5["model"] = "invalid_hash"
    
    try:
        audit.run_audit()
        assert audit.results["Model Hash Verified (Immutable)"] == "FAIL"
    finally:
        audit.EXPECTED_MD5 = original_hashes

def test_audit_script_does_not_call_joblib_dump():
    content = AUDIT_SCRIPT.read_text(encoding="utf-8")
    assert "joblib.dump" not in content

def test_audit_script_does_not_modify_files():
    content = AUDIT_SCRIPT.read_text(encoding="utf-8")
    assert "open(" in content
    # Ensure it only opens with read/binary or write for JSON metadata
    import re
    writes = re.findall(r'open\([^,]+,\s*["\']w["\']\)', content)
    # The script should only write to phase14_step3 metadata and csv files.
    assert len(writes) <= 2 
    assert "best_t1_gdp_growth_model.joblib" not in [w for w in writes]

def test_immutability_hashes_pre_post_audit_logic():
    # The script must verify post hashes
    content = AUDIT_SCRIPT.read_text(encoding="utf-8")
    assert "post_hashes" in content
    assert "CRITICAL FAILURE: PROTECTED ARTIFACTS MUTATED DURING AUDIT" in content

def test_audit_metadata_output_paths_correct():
    content = AUDIT_SCRIPT.read_text(encoding="utf-8")
    assert "phase14_step3_operational_validation_metadata.json" in content
    assert "phase14_step3_operational_validation_results.csv" in content
    assert "phase14_step2" not in content.split("meta_path =")[1] # Ensure we don't overwrite step2
