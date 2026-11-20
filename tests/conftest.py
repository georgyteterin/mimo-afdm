import copy
import numpy as np
import pytest
import phy
from simulation.prepare_sim import derive_params

WAVEFORM_TYPES = list(phy.waveforms.WAVEFORMS)

BASE_CFG = {
    "name": "pytest",
    "mod_order": 4,
    "grid": {"num_sc": 16, "num_blocks": 2, "scs_hz": 15000},
    "channel": {
        "paths": [{"delay_smpl": 0, "avg_gain_db": 0.0}],
        "max_doppler_hz": 0,
        "doppler_mode": "jakes",
        "fading": "none",
    },
    "noise": {"ebn0_db": [0], "count_cp_energy": False},
    "num_trials": 1,
    "seed": 0,
}

# multipath with Doppler inside AFDM / CP limits for num_sc = 16, cp 4 smpl
DOUBLY_DISPERSIVE_CHANNEL = {
    "paths": [
        {"delay_smpl": 0, "avg_gain_db": 0.0, "doppler_norm": 0.0},
        {"delay_smpl": 1, "avg_gain_db": -3.0, "doppler_norm": 0.3},
        {"delay_smpl": 3, "avg_gain_db": -6.0, "doppler_norm": -0.5},
    ],
    "doppler_mode": "fixed",
    "fading": "rayleigh",
}

MULTIPATH_NO_DOPPLER_CHANNEL = {
    "paths": [
        {"delay_smpl": 0, "avg_gain_db": 0.0},
        {"delay_smpl": 2, "avg_gain_db": -3.0},
    ],
    "max_doppler_hz": 0,
    "doppler_mode": "jakes",
    "fading": "rayleigh",
}

def make_cfg(wf_type, equalizer_type="transparent", channel=None, **overrides):
    cfg = copy.deepcopy(BASE_CFG)
    cfg["waveform"] = {"type": wf_type, "cp_len": 0.25}
    cfg["equalizer"] = {"type": equalizer_type}
    if channel is not None:
        cfg["channel"] = copy.deepcopy(channel)
    cfg.update(copy.deepcopy(overrides))
    return derive_params(cfg)

def get_grid_shape(cfg):
    return (cfg["grid"]["num_blocks"], cfg["grid"]["num_sc"])

@pytest.fixture
def rng():
    return np.random.default_rng(12345)
