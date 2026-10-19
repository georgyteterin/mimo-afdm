import numpy as np

def calc_transform_matrices(wf, wf_cfg, num_sc):
    # Unit vectors are passed as blocks, so one call gives matrices for all of them
    block_len_smpl = len(wf.modulate(np.zeros(num_sc, dtype=complex), wf_cfg, (1, num_sc)))

    unit_symbs = np.eye(num_sc, dtype=complex).reshape(-1)
    mod_matrix = wf.modulate(unit_symbs, wf_cfg, (num_sc, num_sc)).reshape(num_sc, block_len_smpl).T

    unit_smpls = np.eye(block_len_smpl, dtype=complex).reshape(-1)
    demod_matrix = wf.demodulate(unit_smpls, wf_cfg, (block_len_smpl, num_sc)).reshape(block_len_smpl, num_sc).T

    return mod_matrix, demod_matrix

def calc_eff_chan_mtrx(chan_implementation, mod_matrix, demod_matrix, num_blocks):
    block_len_smpl, num_sc = mod_matrix.shape
    cp_len_smpl = block_len_smpl - num_sc

    max_delay_smpl = np.max(chan_implementation["delays_smpl"])
    assert max_delay_smpl <= cp_len_smpl, f"Max delay {max_delay_smpl} smpl exceeds cp {cp_len_smpl} smpl: blocks are not independent"

    h_eff = np.zeros((num_blocks, num_sc, num_sc), dtype=complex)
    for block_idx in range(num_blocks):
        time_channel = np.zeros((block_len_smpl, block_len_smpl), dtype=complex)

        for delay_smpl, path_gain, doppler_shift_per_smpl in zip(chan_implementation["delays_smpl"],
                                                               chan_implementation["gains"],
                                                               chan_implementation["doppler_shift_per_smpl"]):
            rx_idxs = np.arange(delay_smpl, block_len_smpl)
            frame_idxs = block_idx * block_len_smpl + rx_idxs
            time_channel[rx_idxs, rx_idxs - delay_smpl] += path_gain * np.exp(2j * np.pi * doppler_shift_per_smpl * frame_idxs)

        h_eff[block_idx] = demod_matrix @ time_channel @ mod_matrix

    return h_eff