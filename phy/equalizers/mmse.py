import numpy as np

def mmse_full(rx_symbs, h_eff, noise_energy):
    num_sc = h_eff.shape[-1]
    h_eff_herm = h_eff.conj().swapaxes(-1, -2)

    # x_hat = H^H * (H * H^H + N0 * I)^-1 * y
    cov_mtrx = h_eff @ h_eff_herm + noise_energy * np.eye(num_sc)
    whitened_rx_symbs = np.linalg.solve(cov_mtrx, rx_symbs[..., None])
    est_symbs_col = h_eff_herm @ whitened_rx_symbs

    est_symbs = est_symbs_col[..., 0]
    return est_symbs

def mmse_diag(rx_symbs, h_eff, noise_energy):
    sc_gains = np.diagonal(h_eff, axis1=-2, axis2=-1)

    # Scalar case of the same formula: x_hat = h^* * (h * h^* + N0)^-1 * y
    cov_mtrx = sc_gains * sc_gains.conj() + noise_energy
    whitened_rx_symbs = rx_symbs / cov_mtrx
    est_symbs = sc_gains.conj() * whitened_rx_symbs

    return est_symbs