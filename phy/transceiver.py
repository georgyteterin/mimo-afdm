from .qam import qam_mapper, qam_demapper
from .waveforms import WAVEFORMS
from .equalizers import EQUALIZERS

def get_grid_shape(cfg):
    return (cfg["grid"]["num_blocks"], cfg["grid"]["num_sc"])

def run_tx_chain(tx_bits, cfg):
    wf_cfg = cfg["waveform"]
    wf = WAVEFORMS[wf_cfg["type"]]

    tx_symbs = qam_mapper(tx_bits, cfg["mod_order"])
    tx_sig = wf.modulate(tx_symbs, wf_cfg, get_grid_shape(cfg))
    return tx_sig

def run_rx_chain(rx_sig, h_eff, noise_energy, cfg):
    wf_cfg = cfg["waveform"]
    wf = WAVEFORMS[wf_cfg["type"]]
    equalizer = EQUALIZERS[cfg["equalizer"]["type"]]
    grid_shape = get_grid_shape(cfg)

    rx_symbs = wf.demodulate(rx_sig, wf_cfg, grid_shape).reshape(grid_shape)
    est_symbs = equalizer(rx_symbs, h_eff, noise_energy)
    rx_bits = qam_demapper(est_symbs.reshape(-1), cfg["mod_order"])
    return rx_bits
