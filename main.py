import simulation as sim

tests = sim.get_tests("tests/first_test.yaml")


for test in tests:
    num_err = sim.run_test(test)

print(1)