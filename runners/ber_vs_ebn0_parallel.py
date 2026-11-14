import argparse
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import numpy as np
import simulation as sim
from simulation.adaptive_grid import make_coarse_grid, find_curve_end, choose_refine_points, is_below_floor
from runners.ber_vs_ebn0 import select_tests, format_separator, format_summary

DEFAULT_CHUNK_NUM_TRIALS = 50
DEFAULT_TARGET_NUM_ERR = 1000   # relative std of BER ~ 1/sqrt(num_err): ~3%
DEFAULT_CPU_FRACTION = 0.9
PROGRESS_LOG_STEP = 0.1        # fraction of Eb/N0 points done
PROGRESS_LOG_INTERVAL_S = 30   # heartbeat when points finish slowly
BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

logger = logging.getLogger(__name__)

def count_available_cores():
    # affinity respects limits set on the process (Linux); Windows has only cpu_count
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1

def calc_default_num_workers(cpu_fraction=DEFAULT_CPU_FRACTION):
    # int() truncates, i.e. floor for positive numbers
    return max(1, int(cpu_fraction * count_available_cores()))

def parse_args():
    parser = argparse.ArgumentParser(description="BER vs Eb/N0 on many cores with early stop and adaptive grid, short log")
    parser.add_argument("--cfg", required=True, help="path to yaml config, e.g. configs/sim/long/fading_test.yaml")
    parser.add_argument("--test", nargs="*", default=[], help="test names from the config; all tests if omitted")
    parser.add_argument("--out-dir", default="results", help="results root, saved as <out-dir>/<cfg path under configs/sim>/<test>.json")
    parser.add_argument("--workers", type=int, default=None, help=f"processes (default: {DEFAULT_CPU_FRACTION:.0%} of available cores)")
    parser.add_argument("--chunk-trials", type=int, default=DEFAULT_CHUNK_NUM_TRIALS, help="trials per task")
    parser.add_argument("--target-err", type=int, default=DEFAULT_TARGET_NUM_ERR,
                        help="stop a point after this many errors; num_trials from yaml is the maximum; 0 disables")
    return parser.parse_args()

def format_duration(duration_s):
    minutes, seconds = divmod(int(round(duration_s)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"

def split_trials(num_trials, chunk_num_trials):
    return [(j_start, min(j_start + chunk_num_trials, num_trials)) for j_start in range(0, num_trials, chunk_num_trials)]

def build_tasks(tests, point_keys, chunk_num_trials):
    # point_key = (j_test, ebn0_db); task = (point_key, j_chunk, j_trial_start, j_trial_stop);
    # ascending Eb/N0 first: low points stop early by error count, and a point below the
    # adaptive floor cancels higher points of its test before they start
    tasks = []
    for point_key in sorted(point_keys, key=lambda point_key: (point_key[1], point_key[0])):
        for j_chunk, (j_start, j_stop) in enumerate(split_trials(tests[point_key[0]]["num_trials"], chunk_num_trials)):
            tasks.append((point_key, j_chunk, j_start, j_stop))
    return tasks

def init_point(num_chunks):
    return {"chunk_num_err": [None] * num_chunks, "chunk_num_trials": [0] * num_chunks,
            "num_chunks_used": 0, "num_err": 0, "num_trials": 0, "is_done": False}

def update_point(point, j_chunk, num_err, num_trials, target_num_err):
    # errors are summed over chunks 0, 1, 2, ... in order, so the stop point
    # does not depend on which worker finished first
    if point["is_done"]:
        return
    point["chunk_num_err"][j_chunk], point["chunk_num_trials"][j_chunk] = num_err, num_trials
    num_chunks = len(point["chunk_num_err"])
    while point["num_chunks_used"] < num_chunks and point["chunk_num_err"][point["num_chunks_used"]] is not None:
        point["num_err"] += point["chunk_num_err"][point["num_chunks_used"]]
        point["num_trials"] += point["chunk_num_trials"][point["num_chunks_used"]]
        point["num_chunks_used"] += 1
        if target_num_err and point["num_err"] >= target_num_err:
            break
    is_target_reached = bool(target_num_err) and point["num_err"] >= target_num_err
    point["is_done"] = is_target_reached or point["num_chunks_used"] == num_chunks

def cancel_point(point_key, futures_by_point):
    return sum(future.cancel() for future in futures_by_point[point_key])

def run_points(pool, tests, point_keys, chunk_num_trials, target_num_err, min_ber_by_test=None):
    # runs a batch of Eb/N0 points with early stop; returns {point_key: point} with cpu_s;
    # for tests in min_ber_by_test, points above one that fell below the floor are skipped:
    # BER decreases with Eb/N0, so they would be cut from the curve anyway
    tasks = build_tasks(tests, point_keys, chunk_num_trials)
    points = {point_key: init_point(len(split_trials(tests[point_key[0]]["num_trials"], chunk_num_trials))) for point_key in point_keys}
    for point in points.values():
        point["cpu_s"], point["num_trials_run"], point["is_skipped"] = 0.0, 0, False
    min_ber_by_test = min_ber_by_test or {}

    futures = {
        pool.submit(sim.run_trial_chunk, tests[point_key[0]], point_key[1], j_start, j_stop): (point_key, j_chunk, j_stop - j_start)
        for point_key, j_chunk, j_start, j_stop in tasks
    }
    futures_by_point = {}
    for future, (point_key, _, _) in futures.items():
        futures_by_point.setdefault(point_key, []).append(future)

    start_s = time.perf_counter()
    num_computed, num_cancelled = 0, 0
    next_progress, last_log_s = PROGRESS_LOG_STEP, 0.0
    for future in as_completed(futures):
        if future.cancelled():
            continue
        num_computed += 1
        point_key, j_chunk, num_trials = futures[future]
        num_err, chunk_s = future.result()
        point = points[point_key]
        point["cpu_s"] += chunk_s
        point["num_trials_run"] += num_trials
        if not point["is_done"]:
            update_point(point, j_chunk, num_err, num_trials, target_num_err)
            if point["is_done"]:
                num_cancelled += cancel_point(point_key, futures_by_point)
                j_test, ebn0_db = point_key
                num_bits = point["num_trials"] * tests[j_test]["num_bits"]
                if j_test in min_ber_by_test and is_below_floor(point["num_err"], point["num_err"] / num_bits, min_ber_by_test[j_test]):
                    for point_key_above, point_above in points.items():
                        if point_key_above[0] == j_test and point_key_above[1] > ebn0_db and not point_above["is_done"]:
                            point_above["is_done"] = point_above["is_skipped"] = True
                            num_cancelled += cancel_point(point_key_above, futures_by_point)

        elapsed_s = time.perf_counter() - start_s
        num_points_done = sum(point_any["is_done"] for point_any in points.values())
        fraction_done = num_points_done / len(points)
        if fraction_done >= next_progress or elapsed_s - last_log_s >= PROGRESS_LOG_INTERVAL_S:
            # early stop only removes tasks, so the task-based ETA is an upper bound
            num_tasks_left = len(futures) - num_cancelled - num_computed
            eta_s = elapsed_s * num_tasks_left / num_computed
            logger.info("progress: points %d/%d (%3.0f%%), tasks done %d, left <= %d, elapsed %s, ETA <= %s",
                        num_points_done, len(points), 100 * fraction_done, num_computed, num_tasks_left,
                        format_duration(elapsed_s), format_duration(eta_s))
            last_log_s = elapsed_s
            while next_progress <= fraction_done:
                next_progress += PROGRESS_LOG_STEP
    return points

def get_adaptive_cfg(test):
    return test["noise"].get("adaptive")

def get_initial_grid(test):
    adaptive_cfg = get_adaptive_cfg(test)
    return make_coarse_grid(adaptive_cfg) if adaptive_cfg else [float(ebn0_db) for ebn0_db in test["noise"]["ebn0_db"]]

def get_curve(points_of_test, adaptive_cfg):
    # sorted computed points of one test; for adaptive tests cut after the curve end
    ebn0_vals_db = sorted(points_of_test)
    num_errs = [points_of_test[ebn0_db]["num_err"] for ebn0_db in ebn0_vals_db]
    bers = [points_of_test[ebn0_db]["num_err"] / (points_of_test[ebn0_db]["num_trials"] * points_of_test[ebn0_db]["num_bits_per_trial"]) for ebn0_db in ebn0_vals_db]
    if adaptive_cfg:
        j_end = find_curve_end(ebn0_vals_db, num_errs, bers, adaptive_cfg["min_ber"])
        ebn0_vals_db, num_errs, bers = ebn0_vals_db[:j_end + 1], num_errs[:j_end + 1], bers[:j_end + 1]
    return ebn0_vals_db, num_errs, bers

def build_stat(test, points_of_test, ebn0_vals_db):
    num_trials = np.array([points_of_test[ebn0_db]["num_trials"] for ebn0_db in ebn0_vals_db])
    stat = {
        "ebn0_db": np.array(ebn0_vals_db),
        "num_err": np.array([points_of_test[ebn0_db]["num_err"] for ebn0_db in ebn0_vals_db], dtype=int),
        "num_bits": num_trials * test["num_bits"],
        "num_trials": num_trials,
        "j_round": np.array([points_of_test[ebn0_db]["j_round"] for ebn0_db in ebn0_vals_db]),
    }
    stat["ber"] = stat["num_err"] / stat["num_bits"]
    return stat

def run_ber_vs_ebn0_parallel(cfg_path, test_names, out_dir, num_workers=None,
                             chunk_num_trials=DEFAULT_CHUNK_NUM_TRIALS, target_num_err=DEFAULT_TARGET_NUM_ERR):
    if num_workers is None:
        num_workers = calc_default_num_workers()
    tests = select_tests(sim.get_tests(cfg_path), test_names)
    max_rounds = max([get_adaptive_cfg(test)["max_rounds"] for test in tests if get_adaptive_cfg(test)] + [0])

    logger.info("%s", format_separator(f"parallel: {cfg_path}"))
    logger.info("%d tests (%d adaptive), %d workers of %d available cores, chunks of <= %d trials, early stop at %s errors per point",
                len(tests), sum(bool(get_adaptive_cfg(test)) for test in tests), num_workers, count_available_cores(),
                chunk_num_trials, target_num_err if target_num_err else "- (off)")
    for test in tests:
        for warning in test.get("warnings", []):
            logger.warning("%s: %s", test["name"], warning)

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    points_by_test = [{} for _ in tests]   # j_test -> {ebn0_db: point}
    pending_by_test = [get_initial_grid(test) for test in tests]
    cpu_skipped_by_test = [0.0] * len(tests)
    trials_skipped_by_test = [0] * len(tests)

    # one BLAS thread per worker; spawn on every OS so the env applies before numpy loads
    for var in BLAS_THREAD_VARS:
        os.environ[var] = "1"
    start_s = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for j_round in range(max_rounds + 1):
            point_keys = [(j_test, ebn0_db) for j_test, pending in enumerate(pending_by_test) for ebn0_db in pending]
            if not point_keys:
                break
            logger.info("round %d/%d: %d new points (%s)", j_round, max_rounds, len(point_keys),
                        ", ".join(f"{test['name']} {len(pending)}" for test, pending in zip(tests, pending_by_test) if pending))
            min_ber_by_test = {j_test: get_adaptive_cfg(test)["min_ber"] for j_test, test in enumerate(tests) if get_adaptive_cfg(test)}
            for point_key, point in run_points(pool, tests, point_keys, chunk_num_trials, target_num_err, min_ber_by_test).items():
                if point["is_skipped"]:
                    cpu_skipped_by_test[point_key[0]] += point["cpu_s"]
                    trials_skipped_by_test[point_key[0]] += point["num_trials_run"]
                    continue
                point["j_round"], point["num_bits_per_trial"] = j_round, tests[point_key[0]]["num_bits"]
                points_by_test[point_key[0]][point_key[1]] = point

            pending_by_test = [[] for _ in tests]
            for j_test, test in enumerate(tests):
                adaptive_cfg = get_adaptive_cfg(test)
                if not adaptive_cfg or j_round == adaptive_cfg["max_rounds"]:
                    continue
                ebn0_vals_db, num_errs, bers = get_curve(points_by_test[j_test], adaptive_cfg)
                new_points_db = [ebn0_db for ebn0_db in choose_refine_points(ebn0_vals_db, num_errs, bers, adaptive_cfg)
                                 if ebn0_db not in points_by_test[j_test]]
                pending_by_test[j_test] = new_points_db
                logger.info("%s: curve end %g dB, %d intervals to refine", test["name"], ebn0_vals_db[-1], len(new_points_db))

    wall_s = time.perf_counter() - start_s
    stats, cpu_by_name, ms_per_trial = {}, {}, {}
    for j_test, test in enumerate(tests):
        points_of_test = points_by_test[j_test]
        ebn0_vals_db, _, _ = get_curve(points_of_test, get_adaptive_cfg(test))
        stat = build_stat(test, points_of_test, ebn0_vals_db)
        # cpu and trials include all work done: discarded chunks, points cut or skipped above the curve end
        cpu_s = sum(point["cpu_s"] for point in points_of_test.values()) + cpu_skipped_by_test[j_test]
        total_trials = sum(point["num_trials_run"] for point in points_of_test.values()) + trials_skipped_by_test[j_test]
        run_info = {"runner": "parallel", "num_workers": num_workers, "chunk_num_trials": chunk_num_trials,
                    "target_num_err": target_num_err, "num_trials": stat["num_trials"], "j_round": stat["j_round"],
                    "cpu_s": round(cpu_s, 3), "ms_per_trial": round(1e3 * cpu_s / max(total_trials, 1), 4)}
        out_path = sim.save_results(cfg_path, test, stat, started_at, wall_s, out_dir, run_info)
        logger.info("test %d/%d %s: %d points, %g..%g dB, trials per point %d..%d -> %s", j_test + 1, len(tests), test["name"],
                    len(ebn0_vals_db), ebn0_vals_db[0], ebn0_vals_db[-1], stat["num_trials"].min(), stat["num_trials"].max(), out_path)
        stats[test["name"]], cpu_by_name[test["name"]], ms_per_trial[test["name"]] = stat, cpu_s, run_info["ms_per_trial"]

    logger.info("%s", format_separator(f"summary: {len(tests)} tests, wall {format_duration(wall_s)}, cpu {format_duration(sum(cpu_by_name.values()))}"))
    logger.info("%s", format_summary(stats, cpu_by_name, time_label="cpu", ms_per_trial=ms_per_trial))
    return stats

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_ber_vs_ebn0_parallel(args.cfg, args.test, args.out_dir, args.workers, args.chunk_trials, args.target_err)
