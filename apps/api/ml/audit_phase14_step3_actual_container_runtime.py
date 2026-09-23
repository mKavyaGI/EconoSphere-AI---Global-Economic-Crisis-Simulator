"""
EconoSphere AI — Phase 14 Step 3: Actual Container Runtime Audit
================================================================
Real-world end-to-end containerized deployment verification script.
"""

import os
import sys
import json
import time
import requests
import hashlib
import subprocess
from pathlib import Path
import yaml
import tempfile
import csv

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = PROJECT_ROOT / "infrastructure" / "docker"
COMPOSE_FILE = DOCKER_DIR / "docker-compose.yml"
DOCKERIGNORE_FILE = PROJECT_ROOT / "apps" / "api" / ".dockerignore"

ARTIFACTS = {
    "model": PROJECT_ROOT / "models" / "phase11" / "best_t1_gdp_growth_model.joblib",
    "manifest": PROJECT_ROOT / "models" / "phase11" / "phase11_production_release_manifest.json",
    "dataset": PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
}

API_BASE = "http://localhost:8000"
API_URL = f"{API_BASE}/api/v1"

EXPECTED_PREDICTION = 2.6995696601100905
TOLERANCE = 1e-12

VALID_PAYLOAD = {
    "country": "USA",
    "year": 2025,
    "exchange_rate_lcu_usd": 1.0,
    "tariff_rate_pct": 2.5,
    "remittances_usd": 1000000000.0,
    "fdi_net_inflow_usd": 5000000000.0,
    "unemployment_pct": 4.0,
    "imports_pct_gdp": 15.0,
    "tax_revenue_pct_gdp": 20.0,
    "exports_pct_gdp": 12.0,
    "interest_rate_pct": 5.0,
    "reserves_usd": 3000000000.0,
    "current_account_pct_gdp": -2.5,
    "inflation_cpi_pct": 2.5,
    "population_total": 330000000.0,
    "gdp_current_usd": 25000000000000.0,
    "gdp_growth_lag1": 2.0,
    "gdp_growth_lag2": 2.2,
    "gdp_growth_lag3": 1.9,
    "inflation_lag1": 2.4,
    "unemployment_lag1": 4.1,
    "exports_lag1": 11.5,
    "imports_lag1": 14.5,
    "gdp_growth_rolling_mean_3": 2.03,
    "gdp_growth_rolling_std_3": 0.15,
    "gdp_growth_rolling_mean_5": 2.1,
    "inflation_rolling_mean_3": 2.3,
    "trade_openness": 27.0,
    "trade_balance_ratio": 0.8,
    "log_gdp_usd": 30.8,
    "log_population": 19.6,
    "gdp_growth_rolling_std_5": 0.2,
    "inflation_rolling_std_3": 0.1
}

results = {
    "Docker Compose Configuration": "FAIL",
    "Production Configuration": "FAIL",
    "API Image Build": "FAIL",
    "Read-Only Model Mount": "FAIL",
    "Read-Only Data Mount": "FAIL",
    "Container Startup": "FAIL",
    "Production Preflight": "FAIL",
    "Health Endpoint": "FAIL",
    "Readiness Endpoint": "FAIL",
    "Production Inference": "FAIL",
    "Prediction Consistency (1e-12)": "FAIL",
    "Candidate B Isolation": "FAIL",
    "Target Leakage Protection": "FAIL",
    "Fail-Closed CORS": "FAIL",
    "Fail-Closed Preflight": "FAIL",
    "Graceful Shutdown": "FAIL",
    "Model Immutability": "FAIL",
    "Dataset Immutability": "FAIL",
    "Manifest Immutability": "FAIL"
}

def get_hash(filepath: Path) -> str:
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
    return md5.hexdigest()

def run_compose_command(args, capture_output=True, env=None, extra_args=None, timeout=300):
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE)]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(args)
    
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
        
    res = subprocess.run(
        cmd,
        cwd=str(DOCKER_DIR),
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=current_env,
        timeout=timeout
    )
    return res

def check_port_8000():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('127.0.0.1', 8000))
        if result == 0:
            print("Port 8000 is in use. Shutting down containers...")
            run_compose_command(["down", "--remove-orphans"])
            time.sleep(2)

def run_audit():
    print("Starting Phase 14 Container Runtime Audit...")
    
    # 1. Pre-audit hashes
    pre_hashes = {k: get_hash(v) for k, v in ARTIFACTS.items()}
    
    check_port_8000()

    try:
        # 2. Docker compose config
        config_res = run_compose_command(["config"])
        if config_res.returncode == 0:
            results["Docker Compose Configuration"] = "PASS"
            parsed = yaml.safe_load(config_res.stdout)
            api_svc = parsed.get("services", {}).get("api", {})
            env_vars = api_svc.get("environment", {})
            if env_vars.get("ENV") == "production" and str(env_vars.get("PRODUCTION_SAFETY_MODE")).lower() == "true":
                results["Production Configuration"] = "PASS"
            
            # Check volumes
            vols = api_svc.get("volumes", [])
            model_ro = False
            data_ro = False
            for v in vols:
                if isinstance(v, dict):
                    target = v.get("target", "")
                    read_only = v.get("read_only", False)
                    if "/models" in target and read_only:
                        model_ro = True
                    if "/data" in target and read_only:
                        data_ro = True
                elif isinstance(v, str):
                    if "/models:ro" in v: model_ro = True
                    if "/data:ro" in v: data_ro = True
                    
            if model_ro: results["Read-Only Model Mount"] = "PASS"
            if data_ro: results["Read-Only Data Mount"] = "PASS"
            
        else:
            print(f"Compose config failed:\n{config_res.stderr}")

        # 3. Build API
        print("Building API image...")
        build_res = run_compose_command(["build", "api"], timeout=120)
        if build_res.returncode == 0:
            results["API Image Build"] = "PASS"
        else:
            print(f"Build failed:\n{build_res.stderr}")
            return

        # 4. Start API
        print("Starting API container...")
        start_res = run_compose_command(["up", "-d", "api"])
        if start_res.returncode == 0:
            results["Container Startup"] = "PASS"
        else:
            print(f"Startup failed:\n{start_res.stderr}")
            return

        # Poll for health with cold-start tolerance
        print("Polling for readiness endpoint...")
        ready = False
        for i in range(60):
            try:
                resp = requests.get(f"{API_URL}/forecasts/readiness", timeout=3)
                if resp.status_code == 200 and resp.json().get("ready") is True:
                    ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        if ready:
            health_data = requests.get(f"{API_URL}/forecasts/health").json()
            if health_data.get("status") == "healthy":
                results["Health Endpoint"] = "PASS"
                results["Production Preflight"] = "PASS"
            
            readiness = requests.get(f"{API_URL}/forecasts/readiness").json()
            if readiness.get("ready") is True:
                results["Readiness Endpoint"] = "PASS"

            # Inference
            resp = requests.post(f"{API_URL}/forecasts/gdp", json=VALID_PAYLOAD)
            if resp.status_code == 200:
                results["Production Inference"] = "PASS"
                pred = resp.json().get("predicted_gdp_growth")
                if pred is not None and abs(pred - EXPECTED_PREDICTION) <= TOLERANCE:
                    results["Prediction Consistency (1e-12)"] = "PASS"

            # Candidate B
            payload_cb = VALID_PAYLOAD.copy()
            payload_cb["growth_regime_num"] = 1.0
            resp_cb = requests.post(f"{API_URL}/forecasts/gdp", json=payload_cb)
            if resp_cb.status_code == 422:
                results["Candidate B Isolation"] = "PASS"

            # Target Leakage
            payload_tl = VALID_PAYLOAD.copy()
            payload_tl["target_year"] = 2026
            resp_tl = requests.post(f"{API_URL}/forecasts/gdp", json=payload_tl)
            if resp_tl.status_code == 422:
                results["Target Leakage Protection"] = "PASS"
        else:
            print("API failed to become ready.")
            logs = run_compose_command(["logs", "api"]).stdout
            print(f"API Logs:\n{logs}")

    finally:
        # Graceful Shutdown
        print("Shutting down API...")
        down_res = run_compose_command(["down", "--remove-orphans"])
        if down_res.returncode == 0:
            results["Graceful Shutdown"] = "PASS"

    # Fail closed CORS test
    print("Testing Wildcard CORS fail-closed...")
    cors_res = subprocess.run(
        [
            "docker", "run", "--rm",
            "-e", "ENV=production",
            "-e", "PRODUCTION_SAFETY_MODE=true",
            "-e", 'ALLOWED_ORIGINS=["*"]',
            "docker-api:latest"
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30
    )
    if cors_res.returncode != 0:
        results["Fail-Closed CORS"] = "PASS"

    # Fail closed preflight test (Mount empty models dir)
    print("Testing Preflight Fail-closed...")
    empty_dir = tempfile.mkdtemp()
    try:
        model_res = subprocess.run(
            [
                "docker", "run", "--rm",
                "-e", "ENV=production",
                "-e", "PRODUCTION_SAFETY_MODE=true",
                "-e", 'ALLOWED_ORIGINS=["https://api.econosphere.ai"]',
                "-v", f"{empty_dir}:/models:ro",
                "docker-api:latest"
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30
        )
        if model_res.returncode != 0:
            results["Fail-Closed Preflight"] = "PASS"
    finally:
        try:
            os.rmdir(empty_dir)
        except Exception:
            pass

    # Post-audit hashes
    post_hashes = {k: get_hash(v) for k, v in ARTIFACTS.items()}
    if pre_hashes["model"] == post_hashes["model"]: results["Model Immutability"] = "PASS"
    if pre_hashes["manifest"] == post_hashes["manifest"]: results["Manifest Immutability"] = "PASS"
    if pre_hashes["dataset"] == post_hashes["dataset"]: results["Dataset Immutability"] = "PASS"

    all_pass = all(v == "PASS" for v in results.values())
    
    print("\n--- PHASE 14 — CONTAINERIZED DEPLOYMENT RESULT ---")
    for k, v in results.items():
        print(f"{k.ljust(35)}: {v}")
    
    print("\nFINAL DECISION:")
    if all_pass:
        print("\nPHASE 14 STEP 2 — FULL CONTAINERIZED RUNTIME VERIFIED")
    else:
        print("\nPHASE 14 NOT VERIFIED")

    # Save outputs
    out_dir = PROJECT_ROOT / "models" / "phase14"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = out_dir / "phase14_step3_containerized_deployment_metadata.json"
    meta = {
        "phase": "14",
        "step": "3",
        "purpose": "Containerized Deployment Validation & Runtime Smoke Testing",
        "production_release_id": "ECONOSPHERE-PHASE11-PROD-2026-08-21",
        "production_status": "PHASE 11 PRODUCTION FROZEN",
    }
    for k, v in results.items():
        meta[k.lower().replace(" ", "_").replace("-", "_")] = v
    meta["overall"] = "PASS" if all_pass else "FAIL"
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    csv_path = out_dir / "phase14_step3_containerized_deployment_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Check", "Status"])
        for k, v in results.items():
            writer.writerow([k, v])

if __name__ == "__main__":
    run_audit()
