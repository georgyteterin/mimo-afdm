from tqdm import tqdm
import logging
import numpy as np
import phy

logger = logging.getLogger(__name__)

def run_test(global_cfg: dict):

    test_name = global_cfg["name"]

    snr_vals = np.array(global_cfg["noise"]["snr_db"])
    num_trials = global_cfg["num_trials"]
    num_bits = global_cfg["num_bits"]

    wf = phy.waveforms.WAVEFORMS[global_cfg["waveform"]["type"]]
    test_cfg = {}
    test_cfg["equ"] = phy.equalizers.EQUALIZERS[global_cfg["equalizer"]["type"]]
    test_cfg["mod_mtrx"], test_cfg["demod_mtrx"] = phy.effective_channel.calc_transform_matrices(wf, global_cfg["waveform"], global_cfg["grid"]["num_sc"])
    

    stat = {}
    stat["ber"] = np.zeros(snr_vals.shape)

    logger.info("%s started: %d SNR points, %d trials each", test_name.upper(), len(snr_vals), num_trials)

    for j_snr, snr_db in enumerate(snr_vals):
        num_err = 0
        for _ in range(num_trials):
            num_err += run_snr_point(snr_db, global_cfg, test_cfg)
        stat["ber"][j_snr] = num_err / (num_trials * num_bits)
        logger.info("[%i/%i] SNR %s dB done: BER=%.3e", j_snr+1, len(snr_vals), snr_db, stat["ber"][j_snr])

    logger.info("%s finished \n", test_name.upper())
    return stat

def run_snr_point(snr_db, global_cfg, test_cfg):

    wf_cfg = global_cfg["waveform"]
    grid_params = global_cfg["grid"]
    num_bits = global_cfg["num_bits"]
    mod_order = global_cfg["mod_order"]
    is_cp_energy_counted = global_cfg["noise"]["count_cp_energy"]

    equalizer = test_cfg["equ"]
    mod_matrix = test_cfg["mod_mtrx"]
    demod_matrix = test_cfg["demod_mtrx"]

    grid_shape = (grid_params["num_blocks"], grid_params["num_sc"])
    wf = phy.waveforms.WAVEFORMS[wf_cfg["type"]]

    # Pipeline
    tx_bits = np.random.randint(2, size=num_bits)
    tx_symbs = phy.qam_mapper(tx_bits, mod_order)

    tx_sig = wf.modulate(tx_symbs, wf_cfg, grid_shape)

    chan_implementation = phy.fading_channel.generate(global_cfg)
    rx_sig = phy.fading_channel.apply(tx_sig, chan_implementation)
    rx_sig, noise_energy = phy.add_noise(rx_sig, snr_db, num_bits, grid_params["num_re"], is_cp_energy_counted)

    rx_symbs = wf.demodulate(rx_sig, wf_cfg, grid_shape).reshape(grid_shape)

    h_eff = phy.effective_channel.calc_eff_chan_mtrx(chan_implementation, mod_matrix, demod_matrix, grid_params["num_blocks"])
    est_symbs = equalizer(rx_symbs, h_eff, noise_energy)
    rx_bits = phy.qam_demapper(est_symbs.reshape(-1), mod_order)

    num_bit_errors = np.sum(tx_bits != rx_bits)
    return num_bit_errors