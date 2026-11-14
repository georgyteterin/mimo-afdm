import json
import logging
import platform
import subprocess
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_CFG_ROOT = REPO_ROOT / "configs" / "sim"

logger = logging.getLogger(__name__)

def get_git_info():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "is_dirty": bool(status.strip())}

def get_result_subdir(cfg_path):
    # configs/sim/long/fading_test.yaml -> long/fading_test; outside configs/sim -> file stem
    cfg_path = Path(cfg_path).resolve()
    try:
        return cfg_path.relative_to(SIM_CFG_ROOT).with_suffix("")
    except ValueError:
        return Path(cfg_path.stem)

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")

def save_results(cfg_path, test_cfg, stat, started_at, duration_s, out_dir="results", run_info=None):
    git_info = get_git_info()
    if git_info is None:
        logger.warning("%s: git info unavailable", test_cfg["name"])
    elif git_info["is_dirty"]:
        logger.warning("%s: uncommitted changes, result is not reproducible from commit %s", test_cfg["name"], git_info["commit"][:7])

    result = {
        "test_name": test_cfg["name"],
        "cfg_path": str(cfg_path),
        "cfg": test_cfg,
        "ebn0_db": stat["ebn0_db"],
        "ber": stat["ber"],
        "num_err": stat["num_err"],
        "num_bits": stat["num_bits"],
        "git": git_info,
        "started_at": started_at,
        "duration_s": round(duration_s, 3),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
    }
    if run_info is not None:
        result["run_info"] = run_info

    out_path = Path(out_dir) / get_result_subdir(cfg_path) / f"{test_cfg['name']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=convert_numpy)
    return out_path
