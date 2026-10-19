import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import simulation as sim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

tests = sim.get_tests("tests/fading_test.yaml")

stat = {}
with logging_redirect_tqdm():
    for test in tests:
        testname = test["name"] 
        stat[testname] = sim.run_test(test)

print(1)