import numpy as np
import source

def run_test(cfg: dict):

    num_err = run_snr_point()

    return num_err

def run_snr_point(): 

    tx_bits = np.random.randint(2, size=16)
    tx_symbs = source.qam_mapper(tx_bits, 4)
    rx_symbs = tx_symbs
    rx_bits = source.qam_demapper(rx_symbs, 4)
    
    num_err = np.sum(tx_bits != rx_bits)

    return num_err