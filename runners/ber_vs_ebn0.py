import argparse
import logging
import time
from datetime import datetime
import simulation as sim

def parse_args():
    parser = argparse.ArgumentParser(description="BER vs Eb/N0 for tests from a yaml config")
    parser.add_argument("--cfg", required=True, help="path to yaml config, e.g. configs/sim/awgn_test.yaml")
    parser.add_argument("--test", nargs="*", default=[], help="test names from the config; all tests if omitted")
    parser.add_argument("--out-dir", default="results", help="results root, saved as <out-dir>/<cfg stem>/<test>.json")
    return parser.parse_args()

def select_tests(tests, test_names):
    if not test_names:
        return tests
    tests_by_name = {test["name"]: test for test in tests}
    unknown_names = [name for name in test_names if name not in tests_by_name]
    if unknown_names:
        raise SystemExit(f"Unknown tests {unknown_names}, available: {list(tests_by_name)}")
    return [tests_by_name[name] for name in test_names]

def run_ber_vs_ebn0(cfg_path, test_names, out_dir):
    tests = select_tests(sim.get_tests(cfg_path), test_names)
    stats = {}
    for test in tests:
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        start_s = time.perf_counter()
        stats[test["name"]] = sim.run_test(test)
        sim.save_results(cfg_path, test, stats[test["name"]], started_at, time.perf_counter() - start_s, out_dir)
    return stats

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_ber_vs_ebn0(args.cfg, args.test, args.out_dir)
