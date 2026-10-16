import numpy as np
import source

def run_test(cfg: dict):

    snr_vals = cfg["snr_db"]
    num_trials = cfg["num_trials"]

    for _, j_snr in enumerate(snr_vals):
        for _ in range(num_trials):
            num_err = run_snr_point(j_snr, cfg)

    return num_err

def run_snr_point(snr_db, cfg):

    wf_cfg = cfg["waveform"]
    wf = source.waveforms.WAVEFORMS[wf_cfg["type"]]
    grid_params = cfg["grid"]
    grid_shape = (grid_params["num_blocks"], grid_params["num_sc"])

    tx_bits = np.random.randint(2, size=cfg["num_bits"])
    tx_symbs = source.qam_mapper(tx_bits, cfg["mod_order"])

    tx_sig = wf.modulate(tx_symbs, wf_cfg, grid_shape)

    rx_sig = tx_sig 

    rx_grid = wf.demodulate(rx_sig, wf_cfg, grid_shape)
    rx_symbs = rx_grid.reshape(-1)
    rx_bits = source.qam_demapper(rx_symbs, cfg["mod_order"])

    return np.sum(tx_bits != rx_bits)