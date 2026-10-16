import numpy as np

def modulate(symbs, wf_cfg, grid_shape):
    return np.zeros(symbs.size, dtype=complex)

def demodulate(rx_sig, wf_cfg, grid_shape):
    return np.zeros(np.prod(grid_shape), dtype=complex)