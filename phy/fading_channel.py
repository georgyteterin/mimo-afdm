import numpy as np

def generate(cfg):
    channel_cfg = cfg["channel"]
    num_sc = cfg["grid"]["num_sc"]

    avg_powers = np.array(channel_cfg["avg_powers"], dtype=float)
    num_paths = len(avg_powers)

    if channel_cfg["fading"] == "rayleigh":
        gains = np.sqrt(avg_powers / 2) * (np.random.randn(num_paths) + 1j * np.random.randn(num_paths))
    else:
        gains = np.sqrt(avg_powers).astype(complex)

    if channel_cfg["doppler_mode"] == "jakes":
        arrival_angles_rad = np.random.uniform(-np.pi, np.pi, num_paths)
        dopplers_norm = channel_cfg["max_doppler_shift_norm"] * np.cos(arrival_angles_rad)
    else:
        dopplers_norm = np.array(channel_cfg["fixed_dopplers_norm"], dtype=float)

    return {
        "delays_smpl": np.array(channel_cfg["delays_smpl"], dtype=int),
        "gains": gains,
        "doppler_shift_per_smpl": dopplers_norm / num_sc,
    }

def apply(tx_sig, realization):
    smpl_idxs = np.arange(len(tx_sig))
    rx_sig = np.zeros(len(tx_sig), dtype=complex)

    for delay_smpl, path_gain, doppler_cyc_per_smpl in zip(realization["delays_smpl"],
                                                           realization["gains"],
                                                           realization["doppler_shift_per_smpl"]):
        delayed_sig = np.zeros(len(tx_sig), dtype=complex)
        delayed_sig[delay_smpl:] = tx_sig[:len(tx_sig) - delay_smpl]
        rx_sig += path_gain * delayed_sig * np.exp(2j * np.pi * doppler_cyc_per_smpl * smpl_idxs)

    return rx_sig