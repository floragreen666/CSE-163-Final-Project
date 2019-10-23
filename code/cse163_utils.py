"""
Flora Tang (1863826), Owen Wang (1831955)
CSE 163 AA

Modified cse163_utils.py that provides support for test.py.
"""


import math
import numpy as np
from numpy.testing import assert_allclose
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")


def check_approx_equals(expected, received):
    """
    Checks received against expected, and returns whether or
    not they match (True if they do, False otherwise).
    If the argument is a float, will do an approximate check.
    If the arugment is a data structure will do an approximate check
    on all of its contents.

    Added Functionality:
    If the argument is a numpy.ndarray, will sort arrays before comparing.
    """
    try:
        if type(expected) == dict:
            # first check that keys match, then check that the
            # values approximately match
            return expected.keys() == received.keys() and \
                all([check_approx_equals(expected[k], received[k])
                     for k in expected.keys()])
        elif type(expected) == list or type(expected) == set:
            # Checks both lists/sets contain the same values
            return len(expected) == len(received) and \
                all([check_approx_equals(v1, v2)
                     for v1, v2 in zip(expected, received)])
        elif type(expected) == float:
            return math.isclose(expected, received, abs_tol=0.001)
        elif type(expected) == np.ndarray:
            assert_allclose(np.sort(expected, axis=None),
                            np.sort(received, axis=None), rtol=1e-9, atol=1e-3)
            return True
        else:
            return expected == received
    except Exception as e:
        print(f'EXCEPTION: Raised when checking check_approx_equals {e}')
        return False


def assert_equals(expected, received):
    """
    Checks received against expected, throws an AssertionError
    if they don't match. If the argument is a float, will do an approximate
    check. If the arugment is a data structure will do an approximate check
    on all of its contents.
    """
    assert check_approx_equals(expected, received), \
        f'Failed: Expected {expected}, but received {received}'
