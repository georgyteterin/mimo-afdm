import argparse
import logging
import simulation as sim

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="BER vs Eb/N0 for tests from a yaml config")
    parser.add_argument("--cfg", required=True, help="path to yaml config, e.g. tests/awgn_test.yaml")
    parser.add_argument("--test", nargs="*", default=[], help="test names from the config; all tests if omitted")
    return parser.parse_args()

def select_tests(tests, test_names):
    if not test_names:
        return tests
    tests_by_name = {test["name"]: test for test in tests}
    unknown_names = [name for name in test_names if name not in tests_by_name]
    if unknown_names:
        raise SystemExit(f"Unknown tests {unknown_names}, available: {list(tests_by_name)}")
    return [tests_by_name[name] for name in test_names]

def run_ber_vs_ebn0(cfg_path, test_names):
    tests = select_tests(sim.get_tests(cfg_path), test_names)
    stats = {}
    for test in tests:
        stats[test["name"]] = sim.run_test(test)
    return stats

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_ber_vs_ebn0(args.cfg, args.test)
