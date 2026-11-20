import logging
from pathlib import Path
import yaml
from simulation.save_results import REPO_ROOT, SIM_CFG_ROOT, get_result_subdir

PLOT_CFG_ROOT = REPO_ROOT / "configs" / "plots"

logger = logging.getLogger(__name__)

def get_plot_cfg_path(cfg_path, sim_cfg_root=SIM_CFG_ROOT, plot_cfg_root=PLOT_CFG_ROOT):
    # configs/sim/local/x.yaml -> configs/plots/local/x.yaml; None outside configs/sim
    cfg_path = Path(cfg_path).resolve()
    try:
        return plot_cfg_root / cfg_path.relative_to(sim_cfg_root)
    except ValueError:
        return None

def format_plot_cfg_template(cfg_name, test_names, results_subdir):
    lines = [
        f"# Plot config for {cfg_name}, created on the first run.",
        "# Edit freely, it is never overwritten. Keys and values: configs/plots/REFERENCE.md",
        "title: null",
        "ci: false",
        "theory: []    # e.g. [qpsk_awgn, qpsk_rayleigh]",
        "curves:",
    ]
    for j_test, test_name in enumerate(test_names):
        lines += [f"  - path: {(results_subdir / test_name).as_posix()}", f"    label: {test_name}"]
        if j_test == 0:
            lines += ['    # color: tab:blue', '    # marker: "o"', '    # ls: "-"']
    return "\n".join(lines) + "\n"

def find_missing_curves(plot_cfg_path, test_names, results_subdir):
    # tests whose result path is not among the curve paths
    with open(plot_cfg_path, encoding="utf-8") as f:
        plot_cfg = yaml.safe_load(f) or {}
    curve_paths = {Path(str(curve.get("path", ""))).with_suffix("").as_posix() for curve in plot_cfg.get("curves") or []}
    return [test_name for test_name in test_names if (results_subdir / test_name).as_posix() not in curve_paths]

def prepare_plot_cfg(cfg_path, test_names, out_dir, sim_cfg_root=SIM_CFG_ROOT, plot_cfg_root=PLOT_CFG_ROOT):
    # create the template, or warn about tests without curves; an existing file is never changed
    plot_cfg_path = get_plot_cfg_path(cfg_path, sim_cfg_root, plot_cfg_root)
    if plot_cfg_path is None:
        return None
    results_subdir = Path(out_dir) / get_result_subdir(cfg_path, sim_cfg_root)
    plot_cfg_name = plot_cfg_path.relative_to(plot_cfg_root.parent.parent).as_posix()

    if plot_cfg_path.exists():
        try:
            missing_tests = find_missing_curves(plot_cfg_path, test_names, results_subdir)
        except (yaml.YAMLError, AttributeError):
            logger.warning("plot config %s is not valid yaml, curves not checked", plot_cfg_name)
            return plot_cfg_path
        if missing_tests:
            logger.warning("plot config %s has no curves for: %s", plot_cfg_name, ", ".join(missing_tests))
        return plot_cfg_path

    plot_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_name = Path(cfg_path).resolve().relative_to(sim_cfg_root.parent.parent).as_posix()
    plot_cfg_path.write_text(format_plot_cfg_template(cfg_name, test_names, results_subdir), encoding="utf-8")
    logger.info("created plot config %s", plot_cfg_name)
    return plot_cfg_path
