import json
from pathlib import Path
import numpy as np
import pytest
import yaml
import simulation as sim
from plotters.plot_ber import check_plot_cfg, calc_clopper_pearson
from conftest import make_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_CFG_PATHS = sorted((REPO_ROOT / "configs" / "sim").glob("*.yaml"))
PLOT_CFG_PATHS = sorted((REPO_ROOT / "configs" / "plots").glob("*.yaml"))

@pytest.mark.parametrize("cfg_path", SIM_CFG_PATHS, ids=lambda path: path.name)
def test_sim_config_runs(cfg_path):
    for cfg in sim.get_tests(cfg_path):
        cfg["num_trials"] = 2
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
