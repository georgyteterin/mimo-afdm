import argparse
import json
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
from matplotlib.lines import Line2D
from scipy.special import erfc
from scipy.stats import beta
from plotters.style import apply_paper_style

CONFIDENCE_LEVEL = 0.95
DEFAULT_COLOR = "tab:blue"
THEORY_LINESTYLES = [(0, (4, 2)), (0, (1, 1)), (0, (6, 2, 1, 2))]
REFERENCE_PATH = "configs/plots/REFERENCE.md"

PLOT_CFG_KEYS = ("title", "ci", "theory", "curves")
CURVE_KEYS = ("path", "label", "color", "marker", "ls", "lw")
CURVE_KEY_TO_KWARG = {"label": "label", "color": "color", "marker": "marker", "ls": "linestyle", "lw": "linewidth"}
LINESTYLES = ("-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted")
MARKERS = tuple(marker for marker in Line2D.markers if isinstance(marker, str) and marker.strip() and marker.lower() != "none")

# Proakis, Digital Communications: Gray-coded QPSK per-bit error probability
def calc_ber_qpsk_awgn(ebn0_db):
    ebn0 = 10 ** (ebn0_db / 10)
    return 0.5 * erfc(np.sqrt(ebn0))

def calc_ber_qpsk_rayleigh(ebn0_db):
    ebn0 = 10 ** (ebn0_db / 10)
    return 0.5 * (1 - np.sqrt(ebn0 / (1 + ebn0)))

THEORY = {
    "qpsk_awgn": (calc_ber_qpsk_awgn, "QPSK AWGN, theory"),
    "qpsk_rayleigh": (calc_ber_qpsk_rayleigh, "QPSK flat Rayleigh, theory"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Plot BER vs Eb/N0 from result JSON files")
    parser.add_argument("paths", nargs="*", default=[], help="quick look: result JSON files, directories or globs")
    parser.add_argument("--plot-cfg", default=None, help=f"yaml with curves and styles, see {REFERENCE_PATH}")
    parser.add_argument("--ci", action="store_true", help=f"quick look: {CONFIDENCE_LEVEL:.0%} Clopper-Pearson intervals")
    parser.add_argument("--theory", nargs="*", default=[], choices=list(THEORY), help="quick look: theoretical curves")
    args = parser.parse_args()
    if bool(args.paths) == bool(args.plot_cfg):
        parser.error("give either result paths or --plot-cfg")
    return args

def resolve_json_path(path_str):
    path = Path(path_str)
    if not path.exists() and path.suffix != ".json" and path.with_suffix(".json").exists():
        return path.with_suffix(".json")
    return path

def collect_json_paths(paths):
    json_paths = []
    for path_str in paths:
        path = resolve_json_path(path_str)
        if path.is_dir():
            json_paths += sorted(path.glob("*.json"))
        elif "*" in path_str:
            # shells like PowerShell pass wildcards unexpanded
            json_paths += sorted(Path().glob(path_str))
        else:
            json_paths.append(path)
    return json_paths

def load_result(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)

def check_curve(curve, where, is_result_checked=True):
    errors = []
    unknown_keys = [key for key in curve if key not in CURVE_KEYS]
    if unknown_keys:
        errors.append(f"unknown keys {unknown_keys}, allowed {list(CURVE_KEYS)}")
    if "path" not in curve:
        errors.append("'path' is required")
    elif is_result_checked and not resolve_json_path(curve["path"]).is_file():
        errors.append(f"result file not found: {curve['path']}")
    if "color" in curve and not is_color_like(curve["color"]):
        errors.append(f"unknown color '{curve['color']}'")
    if "marker" in curve and str(curve["marker"]) not in MARKERS:
        errors.append(f"unknown marker '{curve['marker']}'")
    if "ls" in curve and curve["ls"] not in LINESTYLES:
        errors.append(f"unknown ls '{curve['ls']}', allowed {list(LINESTYLES)}")
    if "lw" in curve and not (isinstance(curve["lw"], (int, float)) and curve["lw"] > 0):
        errors.append(f"lw must be a positive number, got '{curve['lw']}'")
    return [f"{where}: {error}" for error in errors]

def check_plot_cfg(plot_cfg, plot_cfg_path, is_result_checked=True):
    errors = []
    unknown_keys = [key for key in plot_cfg if key not in PLOT_CFG_KEYS]
    if unknown_keys:
        errors.append(f"{plot_cfg_path}: unknown keys {unknown_keys}, allowed {list(PLOT_CFG_KEYS)}")
    unknown_theory = [name for name in plot_cfg.get("theory", []) if name not in THEORY]
    if unknown_theory:
        errors.append(f"{plot_cfg_path}: unknown theory {unknown_theory}, allowed {list(THEORY)}")
    curves = plot_cfg.get("curves") or []
    if not curves:
        errors.append(f"{plot_cfg_path}: 'curves' is empty")
    for j_curve, curve in enumerate(curves):
        errors += check_curve(curve, f"{plot_cfg_path}, curve {j_curve + 1}", is_result_checked)
    if errors:
        raise SystemExit("\n".join(errors + [f"See {REFERENCE_PATH}"]))

def load_plot_cfg(plot_cfg_path):
    with open(plot_cfg_path, encoding="utf-8") as f:
        try:
            plot_cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError as err:
            raise SystemExit(f"{plot_cfg_path}: YAML error: {err}\nQuote marker and ls values, e.g. marker: \"*\", ls: \"-\"")
    check_plot_cfg(plot_cfg, plot_cfg_path)
    curves = []
    for curve in plot_cfg["curves"]:
        style = {CURVE_KEY_TO_KWARG[key]: (str(value) if key == "marker" else value) for key, value in curve.items() if key != "path"}
        curves.append((load_result(resolve_json_path(curve["path"])), style))
    return curves, bool(plot_cfg.get("ci", False)), plot_cfg.get("theory", []), plot_cfg.get("title")

# Clopper & Pearson, Biometrika, 1934
def calc_clopper_pearson(num_err, num_bits, confidence_level=CONFIDENCE_LEVEL):
    alpha = 1 - confidence_level
    ber_low = np.where(num_err > 0, beta.ppf(alpha / 2, num_err, num_bits - num_err + 1), 0.0)
    ber_high = np.where(num_err < num_bits, beta.ppf(1 - alpha / 2, num_err + 1, num_bits - num_err), 1.0)
    return ber_low, ber_high

def plot_ber(curves, is_ci_shown=False, theory_names=(), title=None):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)

    for result, style in curves:
        ebn0_db = np.array(result["ebn0_db"], dtype=float)
        num_err = np.array(result["num_err"])
        num_bits = np.array(result["num_bits"])
        ber = num_err / num_bits
        style = {"color": DEFAULT_COLOR, "label": result["test_name"], **style}

        # zero-error points have no meaning on a log axis
        is_err_observed = num_err > 0
        if is_ci_shown:
            ber_low, ber_high = calc_clopper_pearson(num_err, num_bits)
            yerr = np.vstack([ber - ber_low, ber_high - ber])[:, is_err_observed]
            ax.errorbar(ebn0_db[is_err_observed], ber[is_err_observed], yerr=yerr, capsize=3, **style)
        else:
            ax.plot(ebn0_db[is_err_observed], ber[is_err_observed], **style)

    if theory_names:
        ebn0_all_db = np.concatenate([np.array(result["ebn0_db"], dtype=float) for result, _ in curves])
        ebn0_grid_db = np.linspace(ebn0_all_db.min(), ebn0_all_db.max(), 200)
        for j_theory, theory_name in enumerate(theory_names):
            calc_ber, theory_label = THEORY[theory_name]
            ax.plot(ebn0_grid_db, calc_ber(ebn0_grid_db), color="black", marker="",
                    linestyle=THEORY_LINESTYLES[j_theory % len(THEORY_LINESTYLES)], linewidth=0.9, label=theory_label)

    ax.set_yscale("log")
    ax.set_xlabel(r"$E_b/N_0$, dB")
    ax.set_ylabel("BER")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
    ax.legend()
    if title:
        ax.set_title(title)
    return fig

if __name__ == "__main__":
    args = parse_args()
    if args.plot_cfg:
        plot_ber(*load_plot_cfg(args.plot_cfg))
    else:
        curves = [(load_result(json_path), {}) for json_path in collect_json_paths(args.paths)]
        plot_ber(curves, args.ci, args.theory)
    plt.show()
