import logging
import time
import numpy as np
import phy

STREAM_BITS = 0
STREAM_CHANNEL = 1
STREAM_NOISE = 2
EBN0_KEY_SCALE = 100           # 0.01 dB resolution
EBN0_KEY_OFFSET = 1_000_000    # SeedSequence needs non-negative ints

LOW_STAT_NUM_ERR = 100  # relative std of BER ~ 1/sqrt(num_err): 10% at 100 errors

logger = logging.getLogger(__name__)

def make_rngs(base_seed, ebn0_db, j_trial):
    ebn0_key = int(round(ebn0_db * EBN0_KEY_SCALE)) + EBN0_KEY_OFFSET
    def make_rng(stream_id):
        return np.random.default_rng(np.random.SeedSequence([base_seed, ebn0_key, j_trial, stream_id]))
    return {
        "bits": make_rng(STREAM_BITS),
        "channel": make_rng(STREAM_CHANNEL),
        "noise": make_rng(STREAM_NOISE),
    }

def run_test(global_cfg: dict):

    test_name = global_cfg["name"]

    ebn0_vals_db = np.array(global_cfg["noise"]["ebn0_db"])
    num_trials = global_cfg["num_trials"]
    num_bits = global_cfg["num_bits"]
    equalizer_type = global_cfg["equalizer"]["type"]

    wf = phy.waveforms.WAVEFORMS[global_cfg["waveform"]["type"]]
    test_cfg = {}
    test_cfg["is_h_eff_needed"] = phy.equalizers.IS_H_EFF_NEEDED[equalizer_type]
    if test_cfg["is_h_eff_needed"]:
        test_cfg["mod_mtrx"], test_cfg["demod_mtrx"] = phy.effective_channel.calc_transform_matrices(wf, global_cfg["waveform"], global_cfg["grid"]["num_sc"])

    stat = {}
    stat["ebn0_db"] = ebn0_vals_db
    stat["num_err"] = np.zeros(ebn0_vals_db.shape, dtype=int)
    stat["num_bits"] = np.full(ebn0_vals_db.shape, num_trials * num_bits)

    logger.info("%s: Eb/N0 %g..%g dB (%d pts), %d trials", test_name, ebn0_vals_db.min(), ebn0_vals_db.max(), len(ebn0_vals_db), num_trials)
    for warning in global_cfg.get("warnings", []):
        logger.warning("%s: %s", test_name, warning)

    for j_ebn0, ebn0_db in enumerate(ebn0_vals_db):
        start_s = time.perf_counter()
        num_err = 0
        for j_trial in range(num_trials):
            rngs = make_rngs(global_cfg["seed"], ebn0_db, j_trial)
            num_err += run_trial(ebn0_db, global_cfg, test_cfg, rngs)
        stat["num_err"][j_ebn0] = num_err
        logger.info("%s: [%d/%d] %5.1f dB  BER %.2e  (%d err / %d bit)  %.1f s%s",
                    test_name, j_ebn0 + 1, len(ebn0_vals_db), ebn0_db, num_err / stat["num_bits"][j_ebn0],
                    num_err, stat["num_bits"][j_ebn0], time.perf_counter() - start_s,
                    "  low stat" if num_err < LOW_STAT_NUM_ERR else "")

    stat["ber"] = stat["num_err"] / stat["num_bits"]
    return stat

def run_trial(ebn0_db, global_cfg, test_cfg, rngs):

    grid_params = global_cfg["grid"]
    num_bits = global_cfg["num_bits"]
    is_cp_energy_counted = global_cfg["noise"]["count_cp_energy"]

    tx_bits = rngs["bits"].integers(2, size=num_bits)
    tx_sig = phy.transceiver.run_tx_chain(tx_bits, global_cfg)

    chan_implementation = phy.fading_channel.generate(global_cfg, rngs["channel"])
    rx_sig_no_noise = phy.fading_channel.apply(tx_sig, chan_implementation)
    rx_sig, noise_energy = phy.add_noise(rx_sig_no_noise, ebn0_db, num_bits, grid_params["num_re"], is_cp_energy_counted, rngs["noise"])

    h_eff = None
    if test_cfg["is_h_eff_needed"]:
        h_eff = phy.effective_channel.calc_eff_chan_mtrx(chan_implementation, test_cfg["mod_mtrx"], test_cfg["demod_mtrx"], grid_params["num_blocks"])
    rx_bits = phy.transceiver.run_rx_chain(rx_sig, h_eff, noise_energy, global_cfg)

    num_bit_errors = np.sum(tx_bits != rx_bits)
    return num_bit_errors
