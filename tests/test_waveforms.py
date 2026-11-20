import numpy as np
import pytest
import phy
from conftest import WAVEFORM_TYPES, DOUBLY_DISPERSIVE_CHANNEL, make_cfg, get_grid_shape

def draw_symbs(cfg, rng):
    bits = rng.integers(2, size=cfg["num_bits"])
    return phy.qam_mapper(bits, cfg["mod_order"])

@pytest.mark.parametrize("wf_type", WAVEFORM_TYPES)
def test_modulate_demodulate_roundtrip(wf_type, rng):
    cfg = make_cfg(wf_type, channel=DOUBLY_DISPERSIVE_CHANNEL)
    wf = phy.waveforms.WAVEFORMS[wf_type]
    symbs = draw_symbs(cfg, rng)
    tx_sig = wf.modulate(symbs, cfg["waveform"], get_grid_shape(cfg))
    assert np.allclose(wf.demodulate(tx_sig, cfg["waveform"], get_grid_shape(cfg)), symbs)

@pytest.mark.parametrize("wf_type", WAVEFORM_TYPES)
def test_signal_length_and_energy(wf_type, rng):
    # unitary transform: energy without CP equals symbol energy (Eb/N0 scaling relies on it)
    cfg = make_cfg(wf_type)
    wf = phy.waveforms.WAVEFORMS[wf_type]
    num_blocks, num_sc = get_grid_shape(cfg)
    cp_len_smpl = int(num_sc * cfg["waveform"]["cp_len"])
    symbs = draw_symbs(cfg, rng)
    tx_sig = wf.modulate(symbs, cfg["waveform"], get_grid_shape(cfg))
    assert tx_sig.shape == (num_blocks * (num_sc + cp_len_smpl),)
    sig_no_cp = tx_sig.reshape(num_blocks, num_sc + cp_len_smpl)[:, cp_len_smpl:]
    assert np.isclose(np.sum(np.abs(sig_no_cp) ** 2), np.sum(np.abs(symbs) ** 2))
