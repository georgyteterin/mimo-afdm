import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phy

NUM_SC = 64
CP_LEN = 0.25
TX_SC_IDX = 44  # m0: the only non-zero symbol
# Paths: delay [smpl], normalized Doppler alpha = f_D / scs, gain
PATHS = (
    {"delay_smpl": 0, "doppler_norm": 0.4, "gain": 1.0},
    {"delay_smpl": 3, "doppler_norm": -1.3, "gain": 0.7},
    {"delay_smpl": 6, "doppler_norm": 1.6, "gain": 0.5},
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "seminar")

SLIDE_STYLE = {
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "mathtext.fontset": "stix",
}


def build_channel():
    gains = np.array([path["gain"] for path in PATHS], dtype=complex)
    gains /= np.sqrt(np.sum(np.abs(gains) ** 2))
    return {
        "delays_smpl": np.array([path["delay_smpl"] for path in PATHS]),
        "gains": gains,
        "doppler_shift_per_smpl": np.array([path["doppler_norm"] for path in PATHS]) / NUM_SC,
    }


def build_waveform_cfg(wf):
    cfg = {
        "waveform": {"cp_len": CP_LEN},
        "channel": {
            "max_doppler_shift_norm": max(abs(path["doppler_norm"]) for path in PATHS),
            "delays_smpl": [path["delay_smpl"] for path in PATHS],
        },
        "grid": {"num_sc": NUM_SC},
    }
    for message in wf.derive_params(cfg):
        print("warning:", message)
    return cfg["waveform"]


def calc_eff_matrix(wf, wf_cfg, chan_implementation):
    mod_matrix, demod_matrix = phy.effective_channel.calc_transform_matrices(wf, wf_cfg, NUM_SC)
    return phy.effective_channel.calc_eff_chan_mtrx(chan_implementation, mod_matrix, demod_matrix, 1)[0]


def calc_single_symbol_response(wf, wf_cfg, chan_implementation):
    # Column m0 of H_eff = response of the receiver to one transmitted symbol e_m0
    return np.abs(calc_eff_matrix(wf, wf_cfg, chan_implementation)[:, TX_SC_IDX])


def calc_afdm_path_positions(c1):
    # In this simulator: path p lands at (m0 + alpha_p - 2 N c1 l_p) mod N
    return [np.mod(TX_SC_IDX + path["doppler_norm"] - 2 * NUM_SC * c1 * path["delay_smpl"], NUM_SC) for path in PATHS]


def plot_response(ax, response, title):
    sc_idxs = np.arange(NUM_SC)
    markerline, stemlines, baseline = ax.stem(sc_idxs, response, basefmt=" ")
    plt.setp(stemlines, color="b", lw=1.4)
    plt.setp(markerline, color="b", ms=4)
    ax.axvline(TX_SC_IDX, color="gray", ls="--", lw=1.0)
    ax.text(TX_SC_IDX, 1.02, "$m_0$", color="gray", ha="center", va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlim(-1, NUM_SC)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("$|H_{\\mathrm{eff}}[m, m_0]|$")
    ax.set_title(title, pad=18)
    ax.grid(True, linestyle=":", alpha=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_eff_matrix(ax, h_eff_abs, title):
    ax.imshow(h_eff_abs, cmap="viridis", origin="upper", aspect="equal", vmin=0, vmax=1)
    # highlight column m0 shown on the stem plot
    ax.add_patch(Rectangle((TX_SC_IDX - 0.5, -0.5), 1, NUM_SC, fill=False, ec="r", lw=1.2, ls="--"))
    ticks = np.arange(0, NUM_SC, NUM_SC // 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Вход $m'$")
    ax.set_ylabel("Выход $m$")
    ax.set_title(title)


def label_afdm_paths(ax, response_afdm, c1):
    for j_path, (path, position) in enumerate(zip(PATHS, calc_afdm_path_positions(c1))):
        peak_idx = int(np.round(position)) % NUM_SC
        ax.annotate(f"путь {j_path + 1}\n$l = {path['delay_smpl']},\\ \\alpha = {path['doppler_norm']:g}$",
                    xy=(position, response_afdm[peak_idx]), xytext=(position, 0.93),
                    ha="center", va="top", fontsize=12,
                    arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.8, "ls": ":"})


def plot_matrix_and_response():
    plt.rcParams.update(SLIDE_STYLE)
    chan_implementation = build_channel()
    wf_ofdm, wf_afdm = phy.waveforms.WAVEFORMS["ofdm"], phy.waveforms.WAVEFORMS["afdm"]
    cfg_ofdm, cfg_afdm = build_waveform_cfg(wf_ofdm), build_waveform_cfg(wf_afdm)
    h_ofdm_abs = np.abs(calc_eff_matrix(wf_ofdm, cfg_ofdm, chan_implementation))
    h_afdm_abs = np.abs(calc_eff_matrix(wf_afdm, cfg_afdm, chan_implementation))
    chirp_bins = int(round(2 * NUM_SC * cfg_afdm["c1"]))
    afdm_title = f"AFDM ($2Nc_1 = {chirp_bins}$)"

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 2.3]})
    plot_eff_matrix(axes[0, 0], h_ofdm_abs, "OFDM")
    plot_eff_matrix(axes[1, 0], h_afdm_abs, afdm_title)
    plot_response(axes[0, 1], h_ofdm_abs[:, TX_SC_IDX], "OFDM")
    plot_response(axes[1, 1], h_afdm_abs[:, TX_SC_IDX], afdm_title)
    label_afdm_paths(axes[1, 1], h_afdm_abs[:, TX_SC_IDX], cfg_afdm["c1"])
    axes[0, 1].sharex(axes[1, 1])
    axes[0, 1].tick_params(labelbottom=False)
    axes[1, 1].set_xlabel("Индекс поднесущей / DAFT-бина $m$")
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "eff_matrix_and_response.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_single_symbol_response():
    plt.rcParams.update(SLIDE_STYLE)
    chan_implementation = build_channel()
    wf_ofdm, wf_afdm = phy.waveforms.WAVEFORMS["ofdm"], phy.waveforms.WAVEFORMS["afdm"]
    cfg_ofdm, cfg_afdm = build_waveform_cfg(wf_ofdm), build_waveform_cfg(wf_afdm)

    response_ofdm = calc_single_symbol_response(wf_ofdm, cfg_ofdm, chan_implementation)
    response_afdm = calc_single_symbol_response(wf_afdm, cfg_afdm, chan_implementation)
    chirp_bins = int(round(2 * NUM_SC * cfg_afdm["c1"]))

    fig, (ax_ofdm, ax_afdm) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    plot_response(ax_ofdm, response_ofdm, "OFDM")
    plot_response(ax_afdm, response_afdm, f"AFDM ($2Nc_1 = {chirp_bins}$)")

    label_afdm_paths(ax_afdm, response_afdm, cfg_afdm["c1"])

    ax_afdm.set_xlabel("Индекс поднесущей / DAFT-бина $m$")
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "single_symbol_response.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(plot_single_symbol_response())
    print(plot_matrix_and_response())