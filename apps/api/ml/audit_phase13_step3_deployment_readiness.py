"""
Phase 13 Step 3: Deployment Readiness & Production Integration Audit
"""

import os
import sys
import json
import asyncio
from datetime import datetime
import pandas as pd
import hashlib

# Ensure we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.production_health import _compute_file_hashes, _MODEL_PATH, _DATASET_PATH, _MANIFEST_PATH

def audit_immutability():
    print("\n--- IMMUTABILITY AUDIT ---")
    model_md5, model_sha256 = _compute_file_hashes(_MODEL_PATH)
    data_md5, data_sha256 = _compute_file_hashes(_DATASET_PATH)
    manifest_md5, manifest_sha256 = _compute_file_hashes(_MANIFEST_PATH)
    
    print(f"Model MD5:    {model_md5}")
    print(f"Dataset MD5:  {data_md5}")
    print(f"Manifest MD5: {manifest_md5}")
    
    return {
        "model": {"md5": model_md5, "sha256": model_sha256},
        "dataset": {"md5": data_md5, "sha256": data_sha256},
        "manifest": {"md5": manifest_md5, "sha256": manifest_sha256}
    }

async def audit_fail_closed_preflight():
    print("\n--- FAIL CLOSED PREFLIGHT AUDIT ---")
    
    # 1. Start application with mock config simulating production failure
    print("Testing Preflight with missing Model (simulated) in Production Mode...")
    
    # Temporarily monkey patch settings
    from app.core.config import settings
    original_env = settings.env
    original_safety = settings.production_safety_mode
    
    settings.env = "production"
    settings.production_safety_mode = True
    
    # We will temporarily break the path to force a readiness failure
    import app.services.production_health as ph
    original_model_path = ph._MODEL_PATH
    ph._MODEL_PATH = "/tmp/does-not-exist.joblib"
    
    from app.main import lifespan
    from fastapi import FastAPI
    dummy_app = FastAPI()
    
    try:
        async with lifespan(dummy_app):
            print("ERROR: Application started despite broken integrity!")
            success = False
    except RuntimeError as e:
        if "integrity check failed" in str(e):
            print("SUCCESS: Application safely aborted startup (failed closed).")
            success = True
        else:
            print(f"ERROR: Unexpected exception during startup: {e}")
            success = False
    
    # Restore
    ph._MODEL_PATH = original_model_path
    
    print("\nTesting Preflight with valid config in Production Mode...")
    try:
        async with lifespan(dummy_app):
            print("SUCCESS: Application started successfully with valid integrity.")
            success_valid = True
    except Exception as e:
        print(f"ERROR: Application failed to start with valid integrity: {e}")
        success_valid = False
        
    settings.env = original_env
    settings.production_safety_mode = original_safety
    
    return success and success_valid

def main():
    print("Starting Phase 13 Step 3 Audit...")
    hashes = audit_immutability()
    
    preflight_ok = asyncio.run(audit_fail_closed_preflight())
    
    output_dir = os.path.join(os.path.dirname(__file__), "../../../models/phase13")
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        "audit_timestamp": datetime.utcnow().isoformat() + "Z",
        "step": "Phase 13 Step 3",
        "hashes": hashes,
        "preflight_fail_closed_verified": preflight_ok
    }
    
    meta_path = os.path.join(output_dir, "phase13_step3_deployment_readiness_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nAudit complete. Metadata written to {meta_path}")

if __name__ == "__main__":
    main()
