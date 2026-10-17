import numpy as np
import phy

def run_test(cfg: dict):

    snr_vals = np.array(cfg["noise"]["snr_db"])
    num_trials = cfg["num_trials"]
    num_bits = cfg["num_bits"]

    stat = {}
    stat["ber"] = np.zeros(snr_vals.shape)

    for j_snr, snr_db in enumerate(snr_vals):
        num_err = 0
        for _ in range(num_trials):
            num_err += run_snr_point(snr_db, cfg)
        stat["ber"][j_snr] = num_err / (num_trials * num_bits)

    return stat

def run_snr_point(snr_db, cfg):

    wf_cfg = cfg["waveform"]
    grid_params = cfg["grid"]
    num_bits = cfg["num_bits"]
    mod_order = cfg["mod_order"]
    is_cp_energy_counted = cfg["noise"]["count_cp_energy"]

    grid_shape = (grid_params["num_blocks"], grid_params["num_sc"])
    wf = phy.waveforms.WAVEFORMS[wf_cfg["type"]]

    # Pipeline
    tx_bits = np.random.randint(2, size=num_bits)
    tx_symbs = phy.qam_mapper(tx_bits, mod_order)

    tx_sig = wf.modulate(tx_symbs, wf_cfg, grid_shape)

    rx_sig, _ = phy.add_noise(tx_sig, snr_db, num_bits, grid_params["num_re"], is_cp_energy_counted)

    rx_grid = wf.demodulate(rx_sig, wf_cfg, grid_shape)
    rx_symbs = rx_grid.reshape(-1)
    rx_bits = phy.qam_demapper(rx_symbs, mod_order)

    return np.sum(tx_bits != rx_bits)