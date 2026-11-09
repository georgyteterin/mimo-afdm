import numpy as np
import pytest
import phy
from conftest import WAVEFORM_TYPES, DOUBLY_DISPERSIVE_CHANNEL, make_cfg, get_grid_shape

@pytest.mark.parametrize("wf_type", WAVEFORM_TYPES)
def test_h_eff_matches_signal_path(wf_type, rng):
    # noiseless: demod(channel(mod(x))) must equal h_eff @ x block by block
    cfg = make_cfg(wf_type, channel=DOUBLY_DISPERSIVE_CHANNEL)
    wf = phy.waveforms.WAVEFORMS[wf_type]
    grid_shape = get_grid_shape(cfg)

    symbs = phy.qam_mapper(rng.integers(2, size=cfg["num_bits"]), cfg["mod_order"])
    chan_implementation = phy.fading_channel.generate(cfg, rng)
    rx_sig = phy.fading_channel.apply(wf.modulate(symbs, cfg["waveform"], grid_shape), chan_implementation)
    rx_symbs = wf.demodulate(rx_sig, cfg["waveform"], grid_shape).reshape(grid_shape)

    mod_mtrx, demod_mtrx = phy.effective_channel.calc_transform_matrices(wf, cfg["waveform"], cfg["grid"]["num_sc"])
    h_eff = phy.effective_channel.calc_eff_chan_mtrx(chan_implementation, mod_mtrx, demod_mtrx, grid_shape[0])
    expected_symbs = (h_eff @ symbs.reshape(grid_shape)[..., None])[..., 0]
    assert np.allclose(rx_symbs, expected_symbs)

def test_ofdm_h_eff_diagonal_without_doppler(rng):
    from conftest import MULTIPATH_NO_DOPPLER_CHANNEL
    cfg = make_cfg("ofdm", channel=MULTIPATH_NO_DOPPLER_CHANNEL)
    wf = phy.waveforms.WAVEFORMS["ofdm"]
    mod_mtrx, demod_mtrx = phy.effective_channel.calc_transform_matrices(wf, cfg["waveform"], cfg["grid"]["num_sc"])
    h_eff = phy.effective_channel.calc_eff_chan_mtrx(phy.fading_channel.generate(cfg, rng), mod_mtrx, demod_mtrx, 1)[0]
    assert np.allclose(h_eff - np.diag(np.diag(h_eff)), 0)
