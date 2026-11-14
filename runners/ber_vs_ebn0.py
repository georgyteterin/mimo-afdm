import argparse
import logging
import time
from datetime import datetime
import simulation as sim
from simulation.run_simulation import LOW_STAT_NUM_ERR

SEPARATOR_WIDTH = 72

logger = logging.getLogger(__name__)

def format_separator(title):
    return f" {title} ".center(SEPARATOR_WIDTH, "=")

def parse_args():
    parser = argparse.ArgumentParser(description="BER vs Eb/N0 for tests from a yaml config")
    parser.add_argument("--cfg", required=True, help="path to yaml config, e.g. configs/sim/awgn_test.yaml")
    parser.add_argument("--test", nargs="*", default=[], help="test names from the config; all tests if omitted")
    parser.add_argument("--out-dir", default="results", help="results root, saved as <out-dir>/<cfg path under configs/sim>/<test>.json")
    return parser.parse_args()

def select_tests(tests, test_names):
    if not test_names:
        return tests
    tests_by_name = {test["name"]: test for test in tests}
    unknown_names = [name for name in test_names if name not in tests_by_name]
    if unknown_names:
        raise SystemExit(f"Unknown tests {unknown_names}, available: {list(tests_by_name)}")
    return [tests_by_name[name] for name in test_names]

def format_summary(stats, durations_s, time_label="time"):
    ebn0_cols_db = sorted({float(ebn0_db) for stat in stats.values() for ebn0_db in stat["ebn0_db"]})
    name_width = max(len("test"), *(len(name) for name in stats))
    header = f"{'test':<{name_width}}  {time_label:>7}" + "".join(f"{f'{ebn0_db:g} dB':>10} " for ebn0_db in ebn0_cols_db)
    lines = ["BER:", header]
    is_low_stat_found = False
    for name, stat in stats.items():
        ber_by_ebn0 = {float(ebn0_db): (ber, num_err) for ebn0_db, ber, num_err in zip(stat["ebn0_db"], stat["ber"], stat["num_err"])}
        cells = []
        for ebn0_db in ebn0_cols_db:
            if ebn0_db not in ber_by_ebn0:
                cells.append(f"{'-':>10} ")
                continue
            ber, num_err = ber_by_ebn0[ebn0_db]
            is_low_stat = num_err < LOW_STAT_NUM_ERR
            is_low_stat_found |= is_low_stat
            cells.append(f"{f'{ber:.2e}' + ('*' if is_low_stat else ' '):>11}")
        lines.append(f"{name:<{name_width}}  {durations_s[name]:>5.1f} s" + "".join(cells))
    if is_low_stat_found:
        lines.append(f"* fewer than {LOW_STAT_NUM_ERR} errors")
    return "\n".join(lines)

def run_ber_vs_ebn0(cfg_path, test_names, out_dir):
    tests = select_tests(sim.get_tests(cfg_path), test_names)
    stats, durations_s = {}, {}
    for j_test, test in enumerate(tests):
        logger.info("%s", format_separator(f"test {j_test + 1}/{len(tests)}: {test['name']}"))
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        start_s = time.perf_counter()
        stats[test["name"]] = sim.run_test(test)
        durations_s[test["name"]] = time.perf_counter() - start_s
        out_path = sim.save_results(cfg_path, test, stats[test["name"]], started_at, durations_s[test["name"]], out_dir)
        logger.info("%s: done in %.1f s -> %s", test["name"], durations_s[test["name"]], out_path)
    logger.info("%s", format_separator(f"summary: {len(tests)} tests, {sum(durations_s.values()):.1f} s"))
    logger.info("%s", format_summary(stats, durations_s))
    return stats

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_ber_vs_ebn0(args.cfg, args.test, args.out_dir)
