import os
import numpy as np
import matplotlib.pyplot as plt

import simulation as sim
import phy

# Стилистика графиков для публикации
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "mathtext.fontset": "stix",
})

TESTNAME = "science_week"
YAML_PATH = "tests/" + TESTNAME + ".yaml"
OUT_DIR = os.path.join("visualisers", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def get_configs():
    tests = sim.get_tests(YAML_PATH)
    cfg_ofdm = next(t for t in tests if t["waveform"]["type"] == "ofdm")
    cfg_afdm = next(t for t in tests if t["waveform"]["type"] == "afdm")
    return cfg_ofdm, cfg_afdm


def compute_single_realization_heff(cfg_ofdm, cfg_afdm, seed=42, normalize_power=True):
    num_sc = cfg_afdm["grid"]["num_sc"]
    wf_ofdm = phy.waveforms.WAVEFORMS["ofdm"]
    wf_afdm = phy.waveforms.WAVEFORMS["afdm"]

    mod_ofdm, demod_ofdm = phy.effective_channel.calc_transform_matrices(
        wf_ofdm, cfg_ofdm["waveform"], num_sc
    )
    mod_afdm, demod_afdm = phy.effective_channel.calc_transform_matrices(
        wf_afdm, cfg_afdm["waveform"], num_sc
    )

    np.random.seed(seed)
    chan_impl = phy.fading_channel.generate(cfg_afdm)

    if normalize_power:
        # Нормировка полной энергии реализации канала к 1: sum(|h_p|^2) = 1
        chan_impl["gains"] = chan_impl["gains"] / np.sqrt(
            np.sum(np.abs(chan_impl["gains"]) ** 2)
        )

    h_eff_ofdm = phy.effective_channel.calc_eff_chan_mtrx(
        chan_impl, mod_ofdm, demod_ofdm, 1
    )[0]
    h_eff_afdm = phy.effective_channel.calc_eff_chan_mtrx(
        chan_impl, mod_afdm, demod_afdm, 1
    )[0]

    return h_eff_ofdm, h_eff_afdm


def calc_post_eq_sinr_db(h_eff, sinr_eval_db):
    """
    Расчет несмещенного выходного ОСИШ (Post-EQ SINR) по формуле:
    SINR_k = 1 / [(I + gamma * H_eff^H @ H_eff)^-1]_{k,k} - 1
    """
    num_sc = h_eff.shape[-1]
    gamma = 10.0 ** (sinr_eval_db / 10.0)
    eye_n = np.eye(num_sc, dtype=np.complex128)

    gram = h_eff.conj().T @ h_eff
    err_cov = np.linalg.inv(eye_n + gamma * gram)
    diag_mse = np.real(np.diag(err_cov))

    sinr_lin = np.maximum(1.0 / diag_mse - 1.0, 1e-6)
    return 10.0 * np.log10(sinr_lin)


def find_deep_fade_seed(cfg_ofdm, cfg_afdm, sinr_eval_db=15.0, max_seed=300):
    """
    Поиск реализации канала с единичной суммарной мощностью,
    где лучи дают глубокое частотное замирание в полосе OFDM.
    """
    best_seed = 0
    best_score = -1.0

    for s in range(max_seed):
        h_eff_ofdm, h_eff_afdm = compute_single_realization_heff(
            cfg_ofdm, cfg_afdm, seed=s, normalize_power=True
        )
        sinr_ofdm_db = calc_post_eq_sinr_db(h_eff_ofdm, sinr_eval_db)
        sinr_afdm_db = calc_post_eq_sinr_db(h_eff_afdm, sinr_eval_db)

        afdm_mean = np.mean(sinr_afdm_db)
        ofdm_min = np.min(sinr_ofdm_db)
        score = afdm_mean - ofdm_min

        if afdm_mean > 12.5 and score > best_score:
            best_score = score
            best_seed = s

    return best_seed


def plot_fig1_matrices(cfg_ofdm, cfg_afdm):
    print("1/2 Генерация матриц эффективного канала (а, б)...")
    h_eff_ofdm, h_eff_afdm = compute_single_realization_heff(
        cfg_ofdm, cfg_afdm, seed=15, normalize_power=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), constrained_layout=True)

    for ax, mat, label in zip(axes, [h_eff_ofdm, h_eff_afdm], ["(а)", "(б)"]):
        ax.imshow(np.abs(mat), cmap="viridis", origin="upper", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=12, labelpad=6)

    out_path = os.path.join(OUT_DIR, "fig1_effective_matrices.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def plot_fig2_combined_sinr_and_ber(cfg_ofdm, cfg_afdm, sinr_eval_db=15.0):
    print("2/2 Расчет Post-EQ SINR и BER vs Eb/N0 (в одну строку)...")
    
    # --- Данные для панели (а): Post-EQ SINR по символам ---
    best_seed = find_deep_fade_seed(cfg_ofdm, cfg_afdm, sinr_eval_db=sinr_eval_db)
    print(f"Выбран seed={best_seed} для профиля SINR")

    h_eff_ofdm, h_eff_afdm = compute_single_realization_heff(
        cfg_ofdm, cfg_afdm, seed=best_seed, normalize_power=True
    )

    sinr_ofdm_db = calc_post_eq_sinr_db(h_eff_ofdm, sinr_eval_db)
    sinr_afdm_db = calc_post_eq_sinr_db(h_eff_afdm, sinr_eval_db)

    num_sc = cfg_afdm["grid"]["num_sc"]
    sym_idxs = np.arange(num_sc)

    # --- Данные для панели (б): BER vs Eb/N0 ---
    ebn0_db = np.array(cfg_afdm["noise"]["snr_db"], dtype=float)
    stat_ofdm = sim.run_test(cfg_ofdm)
    stat_afdm = sim.run_test(cfg_afdm)

    ber_ofdm = np.maximum(stat_ofdm["ber"], 1e-6)
    ber_afdm = np.maximum(stat_afdm["ber"], 1e-6)

    # --- Отрисовка двух графиков в одну строку ---
    fig, (ax_sinr, ax_ber) = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)

    # Панель (а): Post-EQ SINR
    ax_sinr.plot(
        sym_idxs,
        sinr_ofdm_db,
        "r-o",
        lw=1.3,
        markersize=3.5,
        alpha=0.85,
        label="OFDM",
    )
    ax_sinr.plot(
        sym_idxs,
        sinr_afdm_db,
        "b-s",
        lw=1.5,
        markersize=3.5,
        label="AFDM",
    )
    ax_sinr.axhline(
        sinr_eval_db,
        color="k",
        linestyle=":",
        lw=1.0,
        label=f"Входное Eb/N0 ({int(sinr_eval_db)} дБ)",
    )
    ax_sinr.set_xlabel("Индекс символа k\n(а)")
    ax_sinr.set_ylabel("Post-EQ SINR, дБ")
    ax_sinr.set_xlim([0, num_sc - 1])
    ax_sinr.set_ylim([2.0, 17.5])
    ax_sinr.grid(True, linestyle=":", alpha=0.7)
    ax_sinr.legend(loc="lower right")

    # Панель (б): BER vs Eb/N0
    ax_ber.semilogy(
        ebn0_db,
        ber_ofdm,
        "r-o",
        lw=1.6,
        markersize=5,
        label="OFDM",
    )
    ax_ber.semilogy(
        ebn0_db,
        ber_afdm,
        "b-s",
        lw=1.6,
        markersize=5,
        label="AFDM",
    )
    ax_ber.set_xlabel("$E_b / N_0$, дБ\n(б)")
    ax_ber.set_ylabel("BER")
    ax_ber.set_ylim([1e-5, 1.0])
    ax_ber.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_ber.legend(loc="lower left")

    out_path = os.path.join(OUT_DIR, "fig2_sinr_and_ber.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    cfg_o, cfg_a = get_configs()
    plot_fig1_matrices(cfg_o, cfg_a)
    plot_fig2_combined_sinr_and_ber(cfg_o, cfg_a, sinr_eval_db=15.0)