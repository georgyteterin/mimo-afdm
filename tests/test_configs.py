import json
from pathlib import Path
import numpy as np
import pytest
import yaml
import simulation as sim
from plotters.plot_ber import check_plot_cfg, calc_clopper_pearson
from conftest import make_cfg
from simulation.adaptive_grid import make_coarse_grid

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_CFG_PATHS = sorted((REPO_ROOT / "configs" / "sim").rglob("*.yaml"))
PLOT_CFG_PATHS = sorted((REPO_ROOT / "configs" / "plots").glob("*.yaml"))

@pytest.mark.parametrize("cfg_path", SIM_CFG_PATHS, ids=lambda path: path.relative_to(REPO_ROOT / "configs" / "sim").as_posix())
def test_sim_config_runs(cfg_path):
    for cfg in sim.get_tests(cfg_path):
        cfg["num_trials"] = 2
        if "ebn0_db" not in cfg["noise"]:  # adaptive-only config: first coarse point
            cfg["noise"]["ebn0_db"] = make_coarse_grid(cfg["noise"]["adaptive"])
        cfg["noise"]["ebn0_db"] = cfg["noise"]["ebn0_db"][:1]
        stat = sim.run_test(cfg)
        assert 0 <= stat["ber"][0] <= 1

@pytest.mark.parametrize("plot_cfg_path", PLOT_CFG_PATHS, ids=lambda path: path.name)
def test_plot_config_valid(plot_cfg_path):
    with open(plot_cfg_path, encoding="utf-8") as f:
        check_plot_cfg(yaml.safe_load(f), plot_cfg_path, is_result_checked=False)

def test_save_results_roundtrip(tmp_path):
    cfg = make_cfg("ofdm", num_trials=2)
    stat = sim.run_test(cfg)
    out_path = sim.save_results("configs/sim/pytest.yaml", cfg, stat, "t", 0.5, tmp_path)
    assert out_path == tmp_path / "pytest" / "pytest.json"
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["num_err"] == stat["num_err"].tolist() and result["cfg"]["seed"] == cfg["seed"]
    assert {"git", "versions", "started_at", "duration_s"} <= set(result)

def test_clopper_pearson_edges():
    # zero errors: upper bound = 1 - (alpha / 2) ** (1 / n), Clopper & Pearson, 1934
    ber_low, ber_high = calc_clopper_pearson(np.array([0, 10]), np.array([1000, 1000]))
    assert ber_low[0] == 0 and np.isclose(ber_high[0], 1 - 0.025 ** (1 / 1000))
    assert ber_low[1] < 0.01 < ber_high[1]

def test_summary_table():
    from runners.ber_vs_ebn0 import format_summary
    stats = {
        "a": {"ebn0_db": np.array([0, 2]), "ber": np.array([1e-1, 1e-3]), "num_err": np.array([500, 20])},
        "bb": {"ebn0_db": np.array([2, 4]), "ber": np.array([1e-2, 1e-4]), "num_err": np.array([300, 150])},
    }
    lines = format_summary(stats, {"a": 1.0, "bb": 2.0}).splitlines()
    assert lines[1].split() == ["test", "time", "0", "dB", "2", "dB", "4", "dB"]
    assert "1.00e-03*" in lines[2] and lines[2].rstrip().endswith("-")
    assert lines[-1].startswith("* fewer than")

@pytest.mark.parametrize("cfg_path, subdir", [
    (REPO_ROOT / "configs" / "sim" / "fading_test.yaml", "fading_test"),
    (REPO_ROOT / "configs" / "sim" / "long" / "fading_test.yaml", "long/fading_test"),
    (Path("elsewhere") / "my_cfg.yaml", "my_cfg"),
])
def test_result_subdir_mirrors_sim_configs(cfg_path, subdir):
    from simulation.save_results import get_result_subdir
    assert get_result_subdir(cfg_path).as_posix() == subdir
