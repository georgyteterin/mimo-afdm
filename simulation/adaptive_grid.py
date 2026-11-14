import numpy as np

ADAPTIVE_KEYS = ("start_db", "stop_db", "num_coarse_pts", "min_step_db", "max_log10_ber_step", "min_ber", "max_rounds")
GRID_DECIMALS = 6

def check_adaptive_cfg(adaptive_cfg):
    missing_keys = [key for key in ADAPTIVE_KEYS if key not in adaptive_cfg]
    unknown_keys = [key for key in adaptive_cfg if key not in ADAPTIVE_KEYS]
    assert not missing_keys and not unknown_keys, f"adaptive: missing {missing_keys}, unknown {unknown_keys}"
    assert adaptive_cfg["start_db"] < adaptive_cfg["stop_db"], "adaptive: start_db must be below stop_db"
    assert adaptive_cfg["num_coarse_pts"] >= 2, "adaptive: num_coarse_pts must be >= 2"
    assert adaptive_cfg["min_step_db"] > 0, "adaptive: min_step_db must be positive"
    assert adaptive_cfg["max_log10_ber_step"] > 0, "adaptive: max_log10_ber_step must be positive"
    assert 0 < adaptive_cfg["min_ber"] < 1, "adaptive: min_ber must be in (0, 1)"
    assert adaptive_cfg["max_rounds"] >= 0, "adaptive: max_rounds must be >= 0"

def snap_to_grid(ebn0_db, adaptive_cfg):
    # common lattice start_db + k * min_step_db: tests share points and seeds
    num_steps = np.round((ebn0_db - adaptive_cfg["start_db"]) / adaptive_cfg["min_step_db"])
    return float(np.round(adaptive_cfg["start_db"] + num_steps * adaptive_cfg["min_step_db"], GRID_DECIMALS))

def make_coarse_grid(adaptive_cfg):
    coarse_db = np.linspace(adaptive_cfg["start_db"], adaptive_cfg["stop_db"], adaptive_cfg["num_coarse_pts"])
    return sorted({snap_to_grid(ebn0_db, adaptive_cfg) for ebn0_db in coarse_db})

def is_below_floor(num_err, ber, min_ber):
    return num_err == 0 or ber < min_ber

def find_curve_end(ebn0_vals_db, num_errs, bers, min_ber):
    # index of the first point with zero errors or BER below min_ber; later points are dropped
    for j_point in range(len(ebn0_vals_db)):
        if is_below_floor(num_errs[j_point], bers[j_point], min_ber):
            return j_point
    return len(ebn0_vals_db) - 1

def choose_refine_points(ebn0_vals_db, num_errs, bers, adaptive_cfg):
    # midpoints of intervals where BER drops faster than max_log10_ber_step decades;
    # the interval before a below-floor end point is refined while BER there is still above min_ber
    new_points_db = []
    for j_point in range(len(ebn0_vals_db) - 1):
        ebn0_left_db, ebn0_right_db = ebn0_vals_db[j_point], ebn0_vals_db[j_point + 1]
        mid_db = snap_to_grid(0.5 * (ebn0_left_db + ebn0_right_db), adaptive_cfg)
        if not ebn0_left_db < mid_db < ebn0_right_db:
            continue
        if is_below_floor(num_errs[j_point], bers[j_point], adaptive_cfg["min_ber"]):
            continue
        if num_errs[j_point + 1] == 0:
            is_refined = True
        else:
            is_refined = abs(np.log10(bers[j_point]) - np.log10(bers[j_point + 1])) > adaptive_cfg["max_log10_ber_step"]
        if is_refined:
            new_points_db.append(mid_db)
    return new_points_db
