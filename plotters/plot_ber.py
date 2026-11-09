import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.stats import beta
from plotters.style import apply_paper_style

CONFIDENCE_LEVEL = 0.95
DEFAULT_COLOR = "tab:blue"
THEORY_LINESTYLES = [(0, (4, 2)), (0, (1, 1)), (0, (6, 2, 1, 2))]
CURVE_STYLE_KEYS = {"color": "color", "marker": "marker", "ls": "linestyle", "lw": "linewidth", "label": "label"}

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
    parser.add_argument("paths", nargs="*", default=[], help="result JSON files, directories or globs; default style")
    parser.add_argument("--curve", nargs="+", action="append", default=[], metavar="PATH [KEY=VALUE]",
                        help=f"one JSON with its own style, keys: {', '.join(CURVE_STYLE_KEYS)}")
    parser.add_argument("--ci", action="store_true", help=f"show {CONFIDENCE_LEVEL:.0%} Clopper-Pearson intervals")
    parser.add_argument("--theory", nargs="*", default=[], choices=list(THEORY), help="theoretical curves")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    if not args.paths and not args.curve:
        parser.error("give result paths or at least one --curve")
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

def parse_curve_style(style_args):
    style = {}
    for style_arg in style_args:
        key, is_sep_found, value = style_arg.partition("=")
        if not is_sep_found or key not in CURVE_STYLE_KEYS:
            raise SystemExit(f"Bad curve style '{style_arg}', expected KEY=VALUE with KEY in {list(CURVE_STYLE_KEYS)}")
        style[CURVE_STYLE_KEYS[key]] = float(value) if key == "lw" else value
    return style

def load_result(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)

def collect_curves(paths, curve_args):
    curves = [(load_result(json_path), {}) for json_path in collect_json_paths(paths)]
    for curve_arg in curve_args:
        json_path = resolve_json_path(curve_arg[0])
        curves.append((load_result(json_path), parse_curve_style(curve_arg[1:])))
    return curves

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
    plot_ber(collect_curves(args.paths, args.curve), args.ci, args.theory, args.title)
    plt.show()
