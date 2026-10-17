import copy
import numpy as np
import yaml

def derive_params(cfg):
    grid = cfg["grid"]
    grid["num_re"] = grid["num_sc"] * grid["num_blocks"] # re = resource elements

    num_bits_per_sym = np.log2(cfg["mod_order"])
    assert num_bits_per_sym % 1 == 0, "Mod order needs to be power 2 of integer"

    cfg["num_bits"] = grid["num_re"] * int(num_bits_per_sym)
    return cfg

def get_tests(path):
    with open(path, encoding="utf-8") as f:
        tests = yaml.safe_load(f)["tests"]
    return [derive_params({"name": name, **copy.deepcopy(cfg)}) for name, cfg in tests.items()]