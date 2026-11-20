import numpy as np
import pytest
from phy.qam import qam_mapper, qam_demapper, QAM_CONSTELLATIONS

@pytest.mark.parametrize("mod_order", list(QAM_CONSTELLATIONS))
@pytest.mark.parametrize("num_symbs", [1, 6, 384])  # not only powers of two
def test_map_demap_roundtrip(mod_order, num_symbs, rng):
    bits = rng.integers(2, size=num_symbs * int(np.log2(mod_order)))
    assert np.array_equal(qam_demapper(qam_mapper(bits, mod_order), mod_order), bits)

@pytest.mark.parametrize("mod_order", list(QAM_CONSTELLATIONS))
def test_constellation_unit_energy(mod_order):
    assert np.isclose(np.mean(np.abs(QAM_CONSTELLATIONS[mod_order]) ** 2), 1.0)

def test_qpsk_gray_mapping():
    # adjacent points differ in one bit: first bit -> sign of I, second -> sign of Q
    symbs = qam_mapper(np.array([0, 0, 0, 1, 1, 0, 1, 1]), 4)
    assert np.array_equal(np.sign(symbs.real), [-1, -1, 1, 1])
    assert np.array_equal(np.sign(symbs.imag), [1, -1, 1, -1])

def test_demapper_rejects_bits():
    with pytest.raises(AssertionError):
        qam_demapper(np.array([0, 1, 1, 0]), 4)
