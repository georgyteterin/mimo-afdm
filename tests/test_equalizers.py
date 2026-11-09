import numpy as np
import pytest
import phy
from conftest import WAVEFORM_TYPES, DOUBLY_DISPERSIVE_CHANNEL, MULTIPATH_NO_DOPPLER_CHANNEL, make_cfg, get_grid_shape

NOISE_ENERGY_TINY = 1e-10

def run_noiseless(cfg, rng):
    wf = phy.waveforms.WAVEFORMS[cfg["waveform"]["type"]]
    grid_shape = get_grid_shape(cfg)
    tx_bits = rng.integers(2, size=cfg["num_bits"])
    chan_implementation = phy.fading_channel.generate(cfg, rng)
    rx_sig = phy.fading_channel.apply(phy.transceiver.run_tx_chain(tx_bits, cfg), chan_implementation)
    mod_mtrx, demod_mtrx = phy.effective_channel.calc_transform_matrices(wf, cfg["waveform"], cfg["grid"]["num_sc"])
    h_eff = phy.effective_channel.calc_eff_chan_mtrx(chan_implementation, mod_mtrx, demod_mtrx, grid_shape[0])
    return tx_bits, rx_sig, h_eff

@pytest.mark.parametrize("wf_type", WAVEFORM_TYPES)
def test_mmse_full_noiseless_no_errors(wf_type, rng):
    cfg = make_cfg(wf_type, "mmse_full", channel=DOUBLY_DISPERSIVE_CHANNEL)
    tx_bits, rx_sig, h_eff = run_noiseless(cfg, rng)
    assert np.array_equal(phy.transceiver.run_rx_chain(rx_sig, h_eff, NOISE_ENERGY_TINY, cfg), tx_bits)

def test_mmse_diag_equals_full_for_diagonal_channel(rng):
    cfg = make_cfg("ofdm", "mmse_full", channel=MULTIPATH_NO_DOPPLER_CHANNEL)
    _, _, h_eff = run_noiseless(cfg, rng)
    rx_symbs = rng.standard_normal(get_grid_shape(cfg)) + 1j * rng.standard_normal(get_grid_shape(cfg))
    noise_energy = 0.3
    est_full = phy.equalizers.EQUALIZERS["mmse_full"](rx_symbs, h_eff, noise_energy)
    est_diag = phy.equalizers.EQUALIZERS["mmse_diag"](rx_symbs, h_eff, noise_energy)
    assert np.allclose(est_full, est_diag)

def test_transparent_returns_input(rng):
    rx_symbs = rng.standard_normal((2, 16)) + 0j
    assert phy.equalizers.EQUALIZERS["transparent"](rx_symbs, None, 1.0) is rx_symbs

def test_h_eff_flag_for_every_equalizer():
    assert set(phy.equalizers.IS_H_EFF_NEEDED) == set(phy.equalizers.EQUALIZERS)
