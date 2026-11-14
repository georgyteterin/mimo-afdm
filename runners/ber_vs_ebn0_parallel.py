import argparse
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import numpy as np
import simulation as sim
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
    parser = argparse.ArgumentParser(description="BER vs Eb/N0 on many cores with early stop, short log")
    parser.add_argument("--cfg", required=True, help="path to yaml config, e.g. configs/sim/fading_long.yaml")
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

def build_tasks(tests, chunk_num_trials):
    # task = (j_test, j_ebn0, j_chunk, j_trial_start, j_trial_stop);
    # chunk-major order: all points advance together, low Eb/N0 stop early and free the pool
    chunks_by_point = {(j_test, j_ebn0): split_trials(test["num_trials"], chunk_num_trials)
                       for j_test, test in enumerate(tests) for j_ebn0 in range(len(test["noise"]["ebn0_db"]))}
    max_num_chunks = max(len(chunks) for chunks in chunks_by_point.values())
    tasks = []
    for j_chunk in range(max_num_chunks):
        for (j_test, j_ebn0), chunks in chunks_by_point.items():
            if j_chunk < len(chunks):
                tasks.append((j_test, j_ebn0, j_chunk, *chunks[j_chunk]))
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

def build_stat(test, points):
    num_bits_per_trial = test["num_bits"]
    stat = {
        "ebn0_db": np.array(test["noise"]["ebn0_db"]),
        "num_err": np.array([point["num_err"] for point in points], dtype=int),
        "num_bits": np.array([point["num_trials"] * num_bits_per_trial for point in points]),
        "num_trials": np.array([point["num_trials"] for point in points]),
    }
    stat["ber"] = stat["num_err"] / stat["num_bits"]
    return stat

def run_ber_vs_ebn0_parallel(cfg_path, test_names, out_dir, num_workers=None,
                             chunk_num_trials=DEFAULT_CHUNK_NUM_TRIALS, target_num_err=DEFAULT_TARGET_NUM_ERR):
    if num_workers is None:
        num_workers = calc_default_num_workers()
    tests = select_tests(sim.get_tests(cfg_path), test_names)
    tasks = build_tasks(tests, chunk_num_trials)
    points = {(j_test, j_ebn0): init_point(len(split_trials(test["num_trials"], chunk_num_trials)))
              for j_test, test in enumerate(tests) for j_ebn0 in range(len(test["noise"]["ebn0_db"]))}

    logger.info("%s", format_separator(f"parallel: {cfg_path}"))
    logger.info("%d tests, %d Eb/N0 points, <= %d tasks of <= %d trials", len(tests), len(points), len(tasks), chunk_num_trials)
    logger.info("%d workers of %d available cores, early stop at %s errors per point",
                num_workers, count_available_cores(), target_num_err if target_num_err else "- (off)")
    for test in tests:
        for warning in test.get("warnings", []):
            logger.warning("%s: %s", test["name"], warning)

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    cpu_times_s = [0.0] * len(tests)
    stats_by_name, cpu_by_name = {}, {}

    # one BLAS thread per worker; spawn on every OS so the env applies before numpy loads
    for var in BLAS_THREAD_VARS:
        os.environ[var] = "1"
    start_s = time.perf_counter()
    num_computed, num_cancelled = 0, 0
    next_progress = PROGRESS_LOG_STEP
    last_log_s = 0.0
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {
            pool.submit(sim.run_trial_chunk, tests[j_test], tests[j_test]["noise"]["ebn0_db"][j_ebn0], j_start, j_stop): (j_test, j_ebn0, j_chunk, j_stop - j_start)
            for j_test, j_ebn0, j_chunk, j_start, j_stop in tasks
        }
        futures_by_point = {}
        for future, (j_test, j_ebn0, _, _) in futures.items():
            futures_by_point.setdefault((j_test, j_ebn0), []).append(future)

        for future in as_completed(futures):
            if future.cancelled():
                continue
            num_computed += 1
            j_test, j_ebn0, j_chunk, num_trials = futures[future]
            num_err, chunk_s = future.result()
            cpu_times_s[j_test] += chunk_s
            point = points[(j_test, j_ebn0)]
            if not point["is_done"]:
                update_point(point, j_chunk, num_err, num_trials, target_num_err)
                if point["is_done"]:
                    num_cancelled += sum(future_left.cancel() for future_left in futures_by_point[(j_test, j_ebn0)])

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

            test = tests[j_test]
            if test["name"] not in stats_by_name and all(points[(j_test, j_point)]["is_done"] for j_point in range(len(test["noise"]["ebn0_db"]))):
                stat = build_stat(test, [points[(j_test, j_point)] for j_point in range(len(test["noise"]["ebn0_db"]))])
                run_info = {"runner": "parallel", "num_workers": num_workers, "chunk_num_trials": chunk_num_trials,
                            "target_num_err": target_num_err, "num_trials": stat["num_trials"], "cpu_s": round(cpu_times_s[j_test], 3)}
                out_path = sim.save_results(cfg_path, test, stat, started_at, elapsed_s, out_dir, run_info)
                logger.info("test %d/%d %s: done at %s, trials per point %d..%d -> %s", j_test + 1, len(tests), test["name"],
                            format_duration(elapsed_s), stat["num_trials"].min(), stat["num_trials"].max(), out_path)
                stats_by_name[test["name"]] = stat

    for j_test, test in enumerate(tests):
        cpu_by_name[test["name"]] = cpu_times_s[j_test]
    wall_s = time.perf_counter() - start_s
    ordered_names = [test["name"] for test in tests]
    logger.info("%s", format_separator(f"summary: {len(tests)} tests, wall {format_duration(wall_s)}, cpu {format_duration(sum(cpu_times_s))}"))
    logger.info("%s", format_summary({name: stats_by_name[name] for name in ordered_names}, cpu_by_name, time_label="cpu"))
    return {name: stats_by_name[name] for name in ordered_names}

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_ber_vs_ebn0_parallel(args.cfg, args.test, args.out_dir, args.workers, args.chunk_trials, args.target_err)
