import numpy as np

QAM_CONSTELLATIONS = {
    4: np.array([-1 + 1j, -1 - 1j, 1 + 1j, 1 - 1j]) / np.sqrt(2),
    }

def qam_mapper(bits: np.ndarray, mod_order: int) -> np.ndarray:

    num_bits_per_symbol = np.log2(mod_order)
    assert len(bits) % num_bits_per_symbol == 0, "Number of bits must be a multiple of log2(mod_order)"
    num_bits_per_symbol = int(num_bits_per_symbol)

    constellation = QAM_CONSTELLATIONS[mod_order]

    num_symbols = len(bits) // num_bits_per_symbol

    assert is_power_of_two(num_symbols), "Number of symbols must be a power of 2"
    
    bit_chunks = bits.reshape(num_symbols, num_bits_per_symbol)
    weights = 2**np.arange(num_bits_per_symbol)[::-1]
    symb_idxs = bit_chunks @ weights

    symbols = constellation[symb_idxs]
    
    return symbols

def qam_demapper(symbs: np.ndarray, mod_order: int) -> np.ndarray:
    
    assert np.iscomplexobj(symbs), "qam_demapper expects complex symbols, not bits"

    constellation = QAM_CONSTELLATIONS[mod_order]
    num_bits_per_symbol = int(np.log2(mod_order))

    dists = np.abs(symbs[:, None] - constellation[None, :])
    symb_idxs = np.argmin(dists, axis=1)

    shifts = np.arange(num_bits_per_symbol)[::-1]
    bits = (symb_idxs[:, None] >> shifts) & 1

    return bits.reshape(-1)

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0