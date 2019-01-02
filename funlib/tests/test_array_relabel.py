from funlib import segment
import numpy as np
import unittest


class TestArrayRelabel(unittest.TestCase):

    def test_replace(self):

        a = np.array([0, 1, 2, 3, 4, 5])
        old = np.array([2, 3])
        new = np.array([20, 30])
        b = segment.arrays.replace_values(a, old, new)

        np.testing.assert_array_equal(a, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(b, [0, 1, 20, 30, 4, 5])

        a += 100
        old = np.array([2, 3, 102, 103])
        new = np.array([20, 30, 2, 3])
        b = segment.arrays.replace_values(a, old, new)

        np.testing.assert_array_equal(a, [100, 101, 102, 103, 104, 105])
        np.testing.assert_array_equal(b, [100, 101, 2, 3, 104, 105])

        a = np.array([1e6, 1e7], dtype=np.uint64)
        old = np.array([1e6], dtype=np.uint64)
        new = np.array([1], dtype=np.uint64)
        b = segment.arrays.replace_values(a, old, new)

        np.testing.assert_array_equal(a, [1e6, 1e7])
        np.testing.assert_array_equal(b, [1, 1e7])

        a = np.array([0, 1e6, 1e7], dtype=np.uint64)
        old = np.array([1e6], dtype=np.uint64)
        new = np.array([1], dtype=np.uint64)
        b = segment.arrays.replace_values(a, old, new)

        np.testing.assert_array_equal(a, [0, 1e6, 1e7])
        np.testing.assert_array_equal(b, [0, 1, 1e7])

    def test_replace_inplace(self):

        a = np.array([0, 1, 2, 3, 4, 5])
        old = np.array([2, 3])
        new = np.array([20, 30])
        b = segment.arrays.replace_values(a, old, new, inplace=True)

        np.testing.assert_array_equal(a, [0, 1, 20, 30, 4, 5])
        np.testing.assert_array_equal(b, [0, 1, 20, 30, 4, 5])

        a = np.array([0, 1, 2, 3, 4, 5]) + 100
        old = np.array([2, 3, 102, 103])
        new = np.array([20, 30, 2, 3])
        b = segment.arrays.replace_values(a, old, new, inplace=True)

        np.testing.assert_array_equal(a, [100, 101, 2, 3, 104, 105])
        np.testing.assert_array_equal(b, [100, 101, 2, 3, 104, 105])

        a = np.array([1e6, 1e7], dtype=np.uint64)
        old = np.array([1e6], dtype=np.uint64)
        new = np.array([1], dtype=np.uint64)
        b = segment.arrays.replace_values(a, old, new, inplace=True)

        np.testing.assert_array_equal(a, [1, 1e7])
        np.testing.assert_array_equal(b, [1, 1e7])

        a = np.array([0, 1e6, 1e7], dtype=np.uint64)
        old = np.array([1e6], dtype=np.uint64)
        new = np.array([1], dtype=np.uint64)
        b = segment.arrays.replace_values(a, old, new, inplace=True)

        np.testing.assert_array_equal(a, [0, 1, 1e7])
        np.testing.assert_array_equal(b, [0, 1, 1e7])
