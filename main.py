import simulation as sim

tests = sim.get_tests("tests\different_doppler.yaml")

stat = {}
for test in tests:
    testname = test["name"] 
    stat[testname] = sim.run_test(test)

print(1)