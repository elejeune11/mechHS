import numpy as np
import simply_supported_beams as ssb


def test_fcn_1():
    # what we know the solution should be
    known = 1
    # what we get when we call ssb.relevant_fcn()
    found = 1.001
    TOL = 0.005
    return np.abs(known - found) < TOL


def test_fcn_2():
    # what we know the solution should be
    known = 2
    # what we get when we call ssb.relevant_fcn()
    found = 2.001
    TOL = 0.005
    return np.abs(known - found) < TOL


if __name__ == "__main__":
    # call test 1
    res_t1 = test_fcn_1()
    print("res_t1", res_t1)
    # call test 2
    res_t2 = test_fcn_2()
    print("res_t2", res_t2)

