import pytest
from simulation.adaptive_grid import (snap_to_grid, make_coarse_grid, find_curve_end,
                                      choose_refine_points, check_adaptive_cfg)

ADAPTIVE_CFG = {"start_db": 0.0, "stop_db": 30.0, "num_coarse_pts": 7, "min_step_db": 0.5,
                "max_log10_ber_step": 0.5, "min_ber": 1e-6, "max_rounds": 4}

def test_coarse_grid_on_lattice():
    assert make_coarse_grid(ADAPTIVE_CFG) == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    assert make_coarse_grid({**ADAPTIVE_CFG, "num_coarse_pts": 4}) == [0.0, 10.0, 20.0, 30.0]
    assert snap_to_grid(7.3, ADAPTIVE_CFG) == 7.5 and snap_to_grid(7.2, ADAPTIVE_CFG) == 7.0

def test_curve_end_at_first_point_below_floor():
    ebn0_vals_db = [0, 5, 10, 15, 20]
    assert find_curve_end(ebn0_vals_db, [900, 500, 40, 0, 0], [1e-1, 1e-2, 1e-4, 0, 0], 1e-6) == 3
    assert find_curve_end(ebn0_vals_db, [900, 500, 40, 5, 1], [1e-1, 1e-2, 1e-4, 1e-7, 1e-8], 1e-6) == 3
    assert find_curve_end(ebn0_vals_db, [900] * 5, [1e-1] * 5, 1e-6) == 4  # floor: nothing to cut

def test_refine_steep_intervals_and_end():
    # 0-5 dB: 0.3 decade (flat), 5-10 dB: 2 decades (steep), 10-15 dB: last nonzero -> zero errors
    ebn0_vals_db = [0.0, 5.0, 10.0, 15.0]
    new_points_db = choose_refine_points(ebn0_vals_db, [500, 300, 3, 0], [1e-1, 5e-2, 5e-4, 0.0], ADAPTIVE_CFG)
    assert new_points_db == [7.5, 12.5]

def test_no_refine_below_min_step():
    assert choose_refine_points([10.0, 10.5], [500, 1], [1e-1, 1e-5], ADAPTIVE_CFG) == []

def test_no_refine_after_min_ber_reached():
    # left point already below min_ber: the end is resolved
    assert choose_refine_points([20.0, 25.0], [2, 0], [5e-7, 0.0], ADAPTIVE_CFG) == []

@pytest.mark.parametrize("bad_cfg", [
    {**ADAPTIVE_CFG, "start_db": 30.0},
    {**ADAPTIVE_CFG, "min_step_db": 0.0},
    {**ADAPTIVE_CFG, "typo": 1},
    {key: value for key, value in ADAPTIVE_CFG.items() if key != "min_ber"},
])
def test_bad_adaptive_cfg_rejected(bad_cfg):
    with pytest.raises(AssertionError):
        check_adaptive_cfg(bad_cfg)
