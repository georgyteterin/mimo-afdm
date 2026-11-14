from pathlib import Path
import numpy as np
import pytest
import yaml
import simulation as sim
from runners.ber_vs_ebn0_parallel import (run_ber_vs_ebn0_parallel, build_tasks, split_trials,
                                          init_point, update_point, calc_default_num_workers)

REPO_ROOT = Path(__file__).resolve().parents[1]

def write_small_cfg(tmp_path, num_trials=7, ebn0_vals_db=(0, 10)):
    with open(REPO_ROOT / "configs" / "sim" / "fading_test.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for test in raw["tests"].values():
        test["num_trials"] = num_trials
        test["noise"]["ebn0_db"] = list(ebn0_vals_db)
    cfg_path = tmp_path / "small.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return cfg_path

def run_early_stop_reference(test, chunk_num_trials, target_num_err):
    # serial reference of the rule: chunks in order, stop at the first chunk reaching the target
    num_err, num_trials = [], []
    for ebn0_db in test["noise"]["ebn0_db"]:
        point_err, point_trials = 0, 0
        for j_start, j_stop in split_trials(test["num_trials"], chunk_num_trials):
            point_err += sim.run_trial_chunk(test, ebn0_db, j_start, j_stop)[0]
            point_trials += j_stop - j_start
            if point_err >= target_num_err:
                break
        num_err.append(point_err)
        num_trials.append(point_trials)
    return np.array(num_err), np.array(num_trials)

def test_build_tasks_covers_all_trials():
    tests = [{"noise": {"ebn0_db": [0, 5]}, "num_trials": 7}]
    tasks = build_tasks(tests, chunk_num_trials=3)
    assert len(tasks) == 6
    assert [(j_start, j_stop) for _, j_ebn0, _, j_start, j_stop in tasks if j_ebn0 == 0] == [(0, 3), (3, 6), (6, 7)]
    assert [task[2] for task in tasks] == [0, 0, 1, 1, 2, 2]  # chunk-major order

def test_update_point_ignores_completion_order():
    chunk_errs = [400, 300, 500, 200]
    point_in_order, point_reversed = init_point(4), init_point(4)
    for j_chunk in range(4):
        update_point(point_in_order, j_chunk, chunk_errs[j_chunk], 10, target_num_err=1000)
    for j_chunk in reversed(range(4)):
        if not point_reversed["is_done"]:
            update_point(point_reversed, j_chunk, chunk_errs[j_chunk], 10, target_num_err=1000)
    for point in (point_in_order, point_reversed):
        assert point["is_done"] and point["num_err"] == 1200 and point["num_trials"] == 30

def test_default_workers_leave_cores_free():
    assert calc_default_num_workers(0.9) >= 1

def test_parallel_matches_serial_without_early_stop(tmp_path):
    # chunks of 3 trials over 7 trials: uneven split must still be bit-exact
    cfg_path = write_small_cfg(tmp_path)
    stats = run_ber_vs_ebn0_parallel(cfg_path, [], tmp_path / "results", num_workers=2, chunk_num_trials=3, target_num_err=0)
    for test in sim.get_tests(cfg_path):
        assert np.array_equal(sim.run_test(test)["num_err"], stats[test["name"]]["num_err"]), test["name"]

@pytest.mark.parametrize("num_workers", [1, 2])
def test_early_stop_matches_reference(tmp_path, num_workers):
    cfg_path = write_small_cfg(tmp_path, num_trials=12, ebn0_vals_db=(0, 6, 12))
    test = sim.get_tests(cfg_path)[2]
    stats = run_ber_vs_ebn0_parallel(cfg_path, [test["name"]], tmp_path / "results", num_workers, chunk_num_trials=2, target_num_err=300)
    num_err_ref, num_trials_ref = run_early_stop_reference(test, chunk_num_trials=2, target_num_err=300)
    assert np.array_equal(stats[test["name"]]["num_err"], num_err_ref)
    assert np.array_equal(stats[test["name"]]["num_trials"], num_trials_ref)
    assert num_trials_ref[0] < num_trials_ref[-1]  # low Eb/N0 point really stopped early
