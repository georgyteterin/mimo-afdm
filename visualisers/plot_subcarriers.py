import os
import numpy as np
import matplotlib.pyplot as plt

# Conceptual continuous-time picture: time in units of T, frequency in units of 1/T
NUM_SC = 32
BANDWIDTH_PER_T = NUM_SC  # B = N / T, subcarrier spacing 1 / T
SHOWN_SC_IDXS = (0, NUM_SC // 4, NUM_SC // 2, 3 * NUM_SC // 4)
SHOWN_SC_LABELS = ("$0$", "$N/4$", "$N/2$", "$3N/4$")
# number of band sweeps per period T: 0 -> OFDM, 1 -> OCDM, 2*N*c1 -> AFDM
WAVEFORM_SWEEPS = {
    "OFDM": 0,
    # "OCDM": 1,
    "AFDM": 3,  # 2*N*c1 = 2*alpha_max + 1 with alpha_max = 1 (Bemani et al. 2023)
}
NUM_TIME_PTS = 8000
ROW_STEP = 2.8
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "seminar")

SLIDE_STYLE = {
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "mathtext.fontset": "stix",
}


def calc_inst_freq(time_t, j_sc, num_sweeps):
    # f_m(t) = (m / T + k t) mod B, k = num_sweeps * B / T
    chirp_rate = num_sweeps * BANDWIDTH_PER_T
    return np.mod(j_sc + chirp_rate * time_t, BANDWIDTH_PER_T)


def calc_subcarrier(time_t, j_sc, num_sweeps):
    # continuous phase = integral of the wrapped instantaneous frequency, taken analytically:
    # each wrap at t_w subtracts B * (t - t_w) from the unwrapped phase m t + k t^2 / 2
    chirp_rate = num_sweeps * BANDWIDTH_PER_T
    phase_cyc = j_sc * time_t + chirp_rate * time_t**2 / 2
    if chirp_rate > 0:
        max_num_wraps = int(np.floor((j_sc + chirp_rate * time_t[-1]) / BANDWIDTH_PER_T))
        wrap_times_t = (np.arange(1, max_num_wraps + 1) * BANDWIDTH_PER_T - j_sc) / chirp_rate
        phase_cyc -= BANDWIDTH_PER_T * np.sum(np.maximum(time_t[:, None] - wrap_times_t[None, :], 0), axis=1)
    return np.exp(2j * np.pi * phase_cyc)


def plot_panel(ax, time_t, num_sweeps, title):
    row_offsets = [-j_row * ROW_STEP for j_row in range(len(SHOWN_SC_IDXS))]
    for offset, j_sc in zip(row_offsets, SHOWN_SC_IDXS):
        subcarrier = calc_subcarrier(time_t, j_sc, num_sweeps)
        is_first = offset == 0
        ax.plot(time_t, subcarrier.real + offset, "b-", lw=1.0, label="Re" if is_first else None)
        ax.plot(time_t, subcarrier.imag + offset, "r--", lw=1.0, label="Im" if is_first else None)

    ax.set_yticks(row_offsets)
    ax.set_yticklabels(SHOWN_SC_LABELS)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "T/2", "T"])
    ax.set_xlim(0, 1)
    ax.set_ylim(row_offsets[-1] - 1.4, 1.4)
    ax.set_xlabel("Время")
    ax.set_title(title)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_subcarrier_families():
    plt.rcParams.update(SLIDE_STYLE)
    time_t = np.linspace(0, 1, NUM_TIME_PTS)

    fig, axes = plt.subplots(1, len(WAVEFORM_SWEEPS), figsize=(10, 5.5), sharey=True)
    for ax, (name, num_sweeps) in zip(axes, WAVEFORM_SWEEPS.items()):
        plot_panel(ax, time_t, num_sweeps, name)
    axes[0].set_ylabel("Номер поднесущей $m$")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "subcarrier_families.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(plot_subcarrier_families())