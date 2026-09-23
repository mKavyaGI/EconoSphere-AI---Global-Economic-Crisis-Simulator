"""
Phase 14 Step 1 Pre-Deployment Snapshot
Calculates and records the cryptographic fingerprints of all protected artifacts.
"""

import os
import json
import hashlib
from datetime import datetime, timezone

def _compute_file_hashes(path: str) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    
    artifacts = {
        "dataset": {
            "path": "data/raw/master_panel.csv"
        },
        "model": {
            "path": "models/phase11/best_t1_gdp_growth_model.joblib"
        },
        "manifest": {
            "path": "models/phase11/phase11_production_release_manifest.json"
        }
    }
    
    snapshot = {
        "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 14",
        "step": "Step 1 Pre-deployment Snapshot",
        "artifacts": {}
    }
    
    for key, info in artifacts.items():
        abs_path = os.path.join(base_dir, info["path"])
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Missing protected artifact: {abs_path}")
            
        md5_hash, sha256_hash = _compute_file_hashes(abs_path)
        stat = os.stat(abs_path)
        
        snapshot["artifacts"][key] = {
            "relative_path": info["path"],
            "md5": md5_hash,
            "sha256": sha256_hash,
            "size_bytes": stat.st_size,
            "last_modified_timestamp": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        }
        
    out_dir = os.path.join(base_dir, "models", "phase14")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase14_step1_predeployment_snapshot.json")
    
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=4)
        
    print(f"Pre-deployment snapshot saved to: {out_path}")
    print(json.dumps(snapshot, indent=2))

if __name__ == "__main__":
    main()
