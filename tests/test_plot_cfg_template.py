import json
import yaml
from plotters.plot_ber import check_plot_cfg, load_plot_cfg
from plotters.plot_cfg_template import prepare_plot_cfg, get_plot_cfg_path

def make_roots(tmp_path):
    sim_cfg_root, plot_cfg_root = tmp_path / "configs" / "sim", tmp_path / "configs" / "plots"
    (sim_cfg_root / "local").mkdir(parents=True)
    return sim_cfg_root, plot_cfg_root

def test_plot_cfg_path_mirrors_sim_cfg(tmp_path):
    sim_cfg_root, plot_cfg_root = make_roots(tmp_path)
    assert get_plot_cfg_path(sim_cfg_root / "local" / "x.yaml", sim_cfg_root, plot_cfg_root) == plot_cfg_root / "local" / "x.yaml"
    assert get_plot_cfg_path(tmp_path / "elsewhere.yaml", sim_cfg_root, plot_cfg_root) is None

def test_template_created_once_and_valid(tmp_path):
    sim_cfg_root, plot_cfg_root = make_roots(tmp_path)
    cfg_path = sim_cfg_root / "local" / "some_test.yaml"
    plot_cfg_path = prepare_plot_cfg(cfg_path, ["ofdm_awgn", "afdm_awgn"], "results", sim_cfg_root, plot_cfg_root)
    plot_cfg = yaml.safe_load(plot_cfg_path.read_text(encoding="utf-8"))
    check_plot_cfg(plot_cfg, plot_cfg_path)
    assert [curve["path"] for curve in plot_cfg["curves"]] == ["results/local/some_test/ofdm_awgn", "results/local/some_test/afdm_awgn"]

    plot_cfg_path.write_text(plot_cfg_path.read_text(encoding="utf-8") + "# my edit\n", encoding="utf-8")
    prepare_plot_cfg(cfg_path, ["ofdm_awgn", "afdm_awgn"], "results", sim_cfg_root, plot_cfg_root)
    assert plot_cfg_path.read_text(encoding="utf-8").endswith("# my edit\n")

def test_warns_about_tests_without_curves(tmp_path, caplog):
    sim_cfg_root, plot_cfg_root = make_roots(tmp_path)
    cfg_path = sim_cfg_root / "local" / "some_test.yaml"
    plot_cfg_path = prepare_plot_cfg(cfg_path, ["ofdm_awgn"], "results", sim_cfg_root, plot_cfg_root)
    text_before = plot_cfg_path.read_text(encoding="utf-8")
    caplog.clear()
    prepare_plot_cfg(cfg_path, ["ofdm_awgn", "afdm_mp"], "results", sim_cfg_root, plot_cfg_root)
    assert "no curves for: afdm_mp" in caplog.text and "ofdm_awgn" not in caplog.text.split("no curves for:")[1]
    assert plot_cfg_path.read_text(encoding="utf-8") == text_before

def test_missing_result_skipped(tmp_path, caplog):
    result = {"test_name": "a", "ebn0_db": [0, 2], "ber": [0.1, 0.01], "num_err": [100, 10], "num_bits": [1000, 1000]}
    (tmp_path / "a.json").write_text(json.dumps(result), encoding="utf-8")
    plot_cfg_path = tmp_path / "plot.yaml"
    plot_cfg_path.write_text(yaml.safe_dump({"curves": [{"path": str(tmp_path / "a")}, {"path": str(tmp_path / "b")}]}), encoding="utf-8")
    curves, _, _, _ = load_plot_cfg(plot_cfg_path)
    assert len(curves) == 1 and "curve 2" in caplog.text and "skipped" in caplog.text
