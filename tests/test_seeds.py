import copy
import numpy as np
import pytest
import phy
import simulation as sim
from simulation.run_simulation import make_rngs
from conftest import WAVEFORM_TYPES, DOUBLY_DISPERSIVE_CHANNEL, make_cfg

def draw_realizations(cfg, ebn0_db, j_trial):
    rngs = make_rngs(cfg["seed"], ebn0_db, j_trial)
    bits = rngs["bits"].integers(2, size=cfg["num_bits"])
    chan_implementation = phy.fading_channel.generate(cfg, rngs["channel"])
    tx_sig = phy.transceiver.run_tx_chain(bits, cfg)
    noise, _ = phy.add_noise(np.zeros_like(tx_sig), ebn0_db, cfg["num_bits"], cfg["grid"]["num_re"], False, rngs["noise"])
    return bits, chan_implementation, noise

def test_run_test_reproducible():
    cfg = make_cfg("ofdm", "mmse_full", channel=DOUBLY_DISPERSIVE_CHANNEL, num_trials=3)
    cfg["noise"]["ebn0_db"] = [0, 5]
    assert np.array_equal(sim.run_test(copy.deepcopy(cfg))["num_err"], sim.run_test(copy.deepcopy(cfg))["num_err"])

def test_same_realizations_across_waveforms():
    cfgs = [make_cfg(wf_type, channel=DOUBLY_DISPERSIVE_CHANNEL) for wf_type in WAVEFORM_TYPES]
    draws = [draw_realizations(cfg, 4.0, 7) for cfg in cfgs]
    for bits, chan_implementation, noise in draws[1:]:
        assert np.array_equal(bits, draws[0][0])
        assert np.array_equal(chan_implementation["gains"], draws[0][1]["gains"])
        assert np.array_equal(chan_implementation["doppler_shift_per_smpl"], draws[0][1]["doppler_shift_per_smpl"])
        assert np.array_equal(noise, draws[0][2])

@pytest.mark.parametrize("ebn0_db, j_trial", [(6.0, 7), (4.0, 8)])
def test_other_point_or_trial_gives_other_realization(ebn0_db, j_trial):
    cfg = make_cfg("ofdm", channel=DOUBLY_DISPERSIVE_CHANNEL)
    gains_ref = draw_realizations(cfg, 4.0, 7)[1]["gains"]
    assert not np.array_equal(draw_realizations(cfg, ebn0_db, j_trial)[1]["gains"], gains_ref)

def test_streams_independent():
    rngs = make_rngs(0, 4.0, 0)
    draws = [rngs[stream].standard_normal(8) for stream in ("bits", "channel", "noise")]
    assert not np.array_equal(draws[0], draws[1]) and not np.array_equal(draws[1], draws[2])

def test_negative_ebn0_key():
    make_rngs(0, -10.0, 0)
