import copy
import yaml 
import numpy as np
import phy
import logging

US_TO_S_RATIO = 1e-6
ROUND_THRESHOLD = 1e-3
FADING_TYPES = ("rayleigh", "none")
DOPPLER_MODES = ("jakes", "fixed")

logger = logging.getLogger(__name__)

def get_tests(path):
    with open(path, encoding="utf-8") as f:
        tests = yaml.safe_load(f)["tests"]
    return [derive_params({"name": name, **copy.deepcopy(cfg)}) for name, cfg in tests.items()]

def derive_params(cfg):
    grid = cfg["grid"]
    grid["num_re"] = grid["num_sc"] * grid["num_blocks"] # re = resource elements

    num_bits_per_sym = np.log2(cfg["mod_order"])
    assert num_bits_per_sym % 1 == 0, "Mod order needs to be power 2 of integer"

    equalizer_type = cfg["equalizer"]["type"]
    assert equalizer_type in phy.equalizers.EQUALIZERS, f"equalizer must be one of {tuple(phy.equalizers.EQUALIZERS)}"

    cfg["num_bits"] = grid["num_re"] * int(num_bits_per_sym)
    derive_channel(cfg)
    wf = phy.waveforms.WAVEFORMS[cfg["waveform"]["type"]]
    for message in wf.derive_params(cfg):
        logger.warning("%s: %s", cfg["name"], message)

    return cfg


def derive_channel(cfg):
    scs_hz = cfg["grid"]["scs_hz"]
    sampling_rate_hz = cfg["grid"]["num_sc"] * scs_hz
    channel_cfg = cfg["channel"]

    assert channel_cfg["fading"] in FADING_TYPES, f"fading must be one of {FADING_TYPES}"
    assert channel_cfg["doppler_mode"] in DOPPLER_MODES, f"doppler_mode must be one of {DOPPLER_MODES}"
    is_doppler_fixed = channel_cfg["doppler_mode"] == "fixed"

    delays_smpl, avg_gains_db, fixed_dopplers_norm = [], [], []
    for path in channel_cfg["paths"]:
        assert ("delay_us" in path) != ("delay_smpl" in path), "Set exactly one of delay_us / delay_smpl"

        if "delay_smpl" in path:
            delay_smpl = path["delay_smpl"]
        else:
            delay_exact_smpl = path["delay_us"] * US_TO_S_RATIO * sampling_rate_hz
            delay_smpl = int(round(delay_exact_smpl))
            if abs(delay_smpl - delay_exact_smpl) > ROUND_THRESHOLD:
                logger.warning("%s: delay %s us = %.3f smpl, rounded to %d", cfg["name"], path["delay_us"], delay_exact_smpl, delay_smpl)
        delays_smpl.append(delay_smpl)
        avg_gains_db.append(path.get("avg_gain_db", 0.0))

        if is_doppler_fixed:
            assert ("doppler_hz" in path) != ("doppler_norm" in path), "Set exactly one of doppler_hz / doppler_norm"
            # doppler_norm = doppler_hz / scs_hz
            fixed_dopplers_norm.append(path["doppler_norm"] if "doppler_norm" in path else path["doppler_hz"] / scs_hz)
        else:
            assert "doppler_hz" not in path and "doppler_norm" not in path, "Per-path doppler requires doppler_mode: fixed"

    if is_doppler_fixed:
        assert "max_doppler_hz" not in channel_cfg and "max_doppler_norm" not in channel_cfg, "max_doppler requires doppler_mode: jakes"
        max_doppler_shift_norm = max(abs(doppler_norm) for doppler_norm in fixed_dopplers_norm)
        channel_cfg["fixed_dopplers_norm"] = [float(doppler_norm) for doppler_norm in fixed_dopplers_norm]
    else:
        assert ("max_doppler_hz" in channel_cfg) != ("max_doppler_norm" in channel_cfg), "Set exactly one of max_doppler_hz / max_doppler_norm"
        max_doppler_shift_norm = channel_cfg["max_doppler_norm"] if "max_doppler_norm" in channel_cfg else channel_cfg["max_doppler_hz"] / scs_hz

    avg_powers = 10 ** (np.array(avg_gains_db) / 10)
    avg_powers = avg_powers / np.sum(avg_powers)

    channel_cfg["delays_smpl"] = delays_smpl
    channel_cfg["avg_powers"] = avg_powers.tolist()
    channel_cfg["max_doppler_shift_norm"] = float(max_doppler_shift_norm)