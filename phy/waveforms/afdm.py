import numpy as np

def derive_c1_c2():
    c1 = 0.0001
    c2 = 0
    return c1, c2

def chirp_diag(c, n):
    return np.exp(-1j * 2 * np.pi * np.mod(c * n**2, 1))

def idaft(S, N, c1, c2):
    n = np.arange(N)
    return np.conj(chirp_diag(c1, n)) * (np.sqrt(N) * np.fft.ifft(np.conj(chirp_diag(c2, n)) * S, axis=-1))

def daft(x, N, c1, c2):
    n = np.arange(N)
    return chirp_diag(c2, n) * (np.fft.fft(chirp_diag(c1, n) * x, axis=-1) / np.sqrt(N))

def modulate(symbs, wf_cfg, grid_shape):
    _, N = grid_shape
    L = int(N * wf_cfg["cp_len"])
    c1, c2 = derive_c1_c2()

    signal = idaft(symbs.reshape(grid_shape), N, c1, c2)

    n = np.arange(-L, 0)
    phase_mult = np.exp(-1j * 2 * np.pi * np.mod(c1 * (N**2 + 2 * N * n), 1))
    cpp = signal[:, N - L:] * phase_mult
    extended_signal = np.hstack([cpp, signal]).reshape(-1)

    return extended_signal

def demodulate(extended_signal, wf_cfg, grid_shape):
    num_blocks, N = grid_shape
    L = int(N * wf_cfg["cp_len"])
    c1, c2 = derive_c1_c2()

    blocks = extended_signal.reshape(num_blocks, N + L)[:, L:]
    signal = daft(blocks, N, c1, c2).reshape(-1)
    
    return signal