import numpy as np
import pytest
from scipy.special import erfc
import simulation as sim
from conftest import WAVEFORM_TYPES, make_cfg

EBN0_VALS_DB = [0, 4, 8]
NUM_TRIALS = 1000
MAX_DEVIATION_SIGMA = 4

@pytest.mark.parametrize("wf_type", WAVEFORM_TYPES)
def test_awgn_ber_matches_qpsk_theory(wf_type):
    # Proakis, Digital Communications: Gray QPSK, Pb = 0.5 * erfc(sqrt(Eb/N0))
    cfg = make_cfg(wf_type, num_trials=NUM_TRIALS, grid={"num_sc": 64, "num_blocks": 8, "scs_hz": 15000})
    cfg["noise"]["ebn0_db"] = EBN0_VALS_DB
    stat = sim.run_test(cfg)

    ber_theory = 0.5 * erfc(np.sqrt(10 ** (np.array(EBN0_VALS_DB) / 10)))
    sigma = np.sqrt(ber_theory * (1 - ber_theory) / stat["num_bits"])
    deviation_sigma = np.abs(stat["ber"] - ber_theory) / sigma
    assert np.all(deviation_sigma < MAX_DEVIATION_SIGMA), f"deviation {deviation_sigma.round(2)} sigma"
