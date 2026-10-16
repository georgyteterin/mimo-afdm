import numpy as np

QAM_CONSTELLATIONS = {
    4: np.array([-1 + 1j, -1 - 1j, 1 + 1j, 1 - 1j]) / np.sqrt(2),
    }

def qam_mapper(bits: np.ndarray, mod_order: int) -> np.ndarray:

    assert np.log2(len(bits)) % 2 == 0, "Number of bits must be a power of 2"

    num_bits_per_symbol = int(np.log2(mod_order))
    assert len(bits) % num_bits_per_symbol == 0, "Number of bits must be a multiple of log2(mod_order)"

    constellation = QAM_CONSTELLATIONS[mod_order]

    num_symbols = len(bits) // num_bits_per_symbol
    
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