import numpy as np

def add_noise(sig, snr_db, num_bits, num_re, count_cp_energy):

    if count_cp_energy:
        bit_energy = len(sig) / num_bits
    else:
        # TODO: num_re -> num_data_re in case of adding pilots
        bit_energy = num_re / num_bits

    noise_energy = bit_energy / 10 ** (snr_db / 10)
    noise = np.sqrt(noise_energy / 2) * (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape))
    return sig + noise, noise_energy