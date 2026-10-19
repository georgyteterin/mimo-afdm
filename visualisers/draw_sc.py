import numpy as np
import matplotlib.pyplot as plt
import simulation as sim

SPS = 32
SUBCARRIER_IDXS = [0]
BANDWIDTH_CYC_PER_SMPL = 1.0
FREQ_JUMP_THRESHOLD = 0.5

def calc_inst_freq(subcarrier_idx, time_smpl, num_sc, c1):
    inst_freq = 2 * c1 * time_smpl + subcarrier_idx / num_sc
    return inst_freq

def wrap_freq(inst_freq, bandwidth):
    half_bandwidth = bandwidth / 2
    wrapped_freq = np.mod(inst_freq + half_bandwidth, bandwidth) - half_bandwidth
    return wrapped_freq

def calc_chirp(subcarrier_idx, time_smpl, num_sc, c1, c2):
    phase_cycles = c1 * time_smpl ** 2 + c2 * subcarrier_idx ** 2 + time_smpl * subcarrier_idx / num_sc
    chirp = np.exp(2j * np.pi * phase_cycles) / np.sqrt(num_sc)
    return chirp

def calc_wrapped_chirp(subcarrier_idx, time_smpl, num_sc, c1, c2):
    time_step_smpl = time_smpl[1] - time_smpl[0]
    inst_freq = calc_inst_freq(subcarrier_idx, time_smpl, num_sc, c1)
    wrapped_freq = wrap_freq(inst_freq, BANDWIDTH_CYC_PER_SMPL)

    # Phase is the integral of the wrapped frequency, started at the analytic phase
    start_phase_cycles = c2 * subcarrier_idx ** 2
    phase_cycles = start_phase_cycles + np.cumsum(wrapped_freq) * time_step_smpl
    wrapped_chirp = np.exp(2j * np.pi * phase_cycles) / np.sqrt(num_sc)
    return wrapped_chirp

tests = sim.get_tests("tests/fading_test.yaml")
cfg = next(test for test in tests if test["waveform"]["type"] == "afdm")

num_sc = cfg["grid"]["num_sc"]
c1, c2 = cfg["waveform"]["c1"], cfg["waveform"]["c2"]

time_smpl = np.arange(num_sc * SPS) / SPS

fig, (ax_sig, ax_freq) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

for color_idx, subcarrier_idx in enumerate(SUBCARRIER_IDXS):
    color = f"C{color_idx}"

    chirp = calc_chirp(subcarrier_idx, time_smpl, num_sc, c1, c2)
    wrapped_chirp = calc_wrapped_chirp(subcarrier_idx, time_smpl, num_sc, c1, c2)

    inst_freq = calc_inst_freq(subcarrier_idx, time_smpl, num_sc, c1)
    wrapped_freq = wrap_freq(inst_freq, BANDWIDTH_CYC_PER_SMPL)
    wrapped_freq[np.abs(np.diff(wrapped_freq, prepend=wrapped_freq[0])) > FREQ_JUMP_THRESHOLD] = np.nan

    # ax_sig.plot(time_smpl, chirp.real, color=color, alpha=0.25, linewidth=0.8)
    ax_sig.plot(time_smpl, wrapped_chirp.real, '-', color=color, label=f"Re")
    ax_sig.plot(time_smpl, wrapped_chirp.imag, '-.', color="Red", alpha=0.5, label=f"Im")
    # ax_freq.plot(time_smpl, inst_freq, color=color, alpha=0.25, linewidth=0.8)
    ax_freq.plot(time_smpl, wrapped_freq, color=color)

for band_edge in (-BANDWIDTH_CYC_PER_SMPL / 2, BANDWIDTH_CYC_PER_SMPL / 2):
    ax_freq.axhline(band_edge, color="k", linestyle="--", linewidth=0.8)

ax_sig.set_ylabel("Re (faint: analytic chirp)")
ax_freq.set_ylabel("inst. freq, cyc/smpl")
ax_freq.set_xlabel("time, smpl")
ax_sig.legend()
plt.show()