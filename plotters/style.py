import matplotlib.pyplot as plt
from cycler import cycler

PAPER_RC_PARAMS = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "mathtext.fontset": "stix",
    "lines.markersize": 5,
}

# Default: only markers change; color and line style are set per curve if needed
MARKER_CYCLE = cycler(marker=["o", "s", "^", "v", "D", "x", "*", "p", "<", ">"])

def apply_paper_style():
    plt.rcParams.update(PAPER_RC_PARAMS)
    plt.rcParams["axes.prop_cycle"] = MARKER_CYCLE
