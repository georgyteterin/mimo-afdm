import numpy as np

def derive_params(cfg):
    return []

def modulate(symbs, wf_cfg, grid_shape):
    _, num_sc = grid_shape
    sig_to_cp_ratio = wf_cfg["cp_len"]
    cp_len = int(num_sc * sig_to_cp_ratio)

    symbs_on_grid = symbs.reshape(grid_shape)
    signal = np.fft.ifft(symbs_on_grid, axis=1) * np.sqrt(num_sc)
    signal_ext = np.hstack([signal[:, num_sc - cp_len:], signal])
    return signal_ext.reshape(-1)

def demodulate(rx_sig, wf_cfg, grid_shape):
    num_blocks, num_sc = grid_shape
    sig_to_cp_ratio = wf_cfg["cp_len"]
    cp_len = int(num_sc * sig_to_cp_ratio)

    blocks = rx_sig.reshape(num_blocks, num_sc + cp_len)[:, cp_len:]
    grid = np.fft.fft(blocks, axis=1) / np.sqrt(num_sc)
    return grid.reshape(-1)