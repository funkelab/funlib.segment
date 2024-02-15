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

        a = np.array([1e6, 1e12], dtype=np.uint64)
        old = np.array([1e6], dtype=np.uint64)
        new = np.array([1], dtype=np.uint64)
        b = segment.arrays.replace_values(a, old, new)

        np.testing.assert_array_equal(a, [1e6, 1e12])
        np.testing.assert_array_equal(b, [1, 1e12])

        a = np.array([0, 1e6, 1e12], dtype=np.uint64)
        old = np.array([1e6], dtype=np.uint64)
        new = np.array([1], dtype=np.uint64)
        b = segment.arrays.replace_values(a, old, new)

        np.testing.assert_array_equal(a, [0, 1e6, 1e12])
        np.testing.assert_array_equal(b, [0, 1, 1e12])

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

        a = np.array([1e6, 1e12], dtype=np.uint64)
        old = np.array([1e6], dtype=np.uint64)
        new = np.array([1], dtype=np.uint64)
        b = segment.arrays.replace_values(a, old, new, inplace=True)

        np.testing.assert_array_equal(a, [1, 1e12])
        np.testing.assert_array_equal(b, [1, 1e12])

        a = np.array([0, 1e6, 1e12], dtype=np.uint64)
        old = np.array([1e6], dtype=np.uint64)
        new = np.array([1], dtype=np.uint64)
        b = segment.arrays.replace_values(a, old, new, inplace=True)

        np.testing.assert_array_equal(a, [0, 1, 1e12])
        np.testing.assert_array_equal(b, [0, 1, 1e12])

    def test_replace_explicit_output(self):
        a = np.array([0, 1, 2, 3, 4, 5])
        b = a.copy()
        old = np.array([2, 3])
        new = np.array([20, 30])
        b = segment.arrays.replace_values(a, old, new, b)
        np.testing.assert_array_equal(a, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(b, [0, 1, 20, 30, 4, 5])

        # test when b is not reassigned with replace_values
        a = np.array([0, 1, 2, 3, 4, 5])
        b = a.copy()
        old = np.array([2, 3])
        new = np.array([20, 30])
        segment.arrays.replace_values(a, old, new, b)
        np.testing.assert_array_equal(a, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(b, [0, 1, 20, 30, 4, 5])

        # test when b is not a copy of a
        a = np.array([0, 1, 2, 3, 4, 5])
        b = np.array([0, 10, 22, 33, 40, 50])
        old = np.array([2, 3])
        new = np.array([20, 30])
        segment.arrays.replace_values(a, old, new, b)
        np.testing.assert_array_equal(a, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(b, [0, 10, 20, 30, 40, 50])

        a += 100
        old = np.array([2, 3, 102, 103])
        new = np.array([20, 30, 2, 3])
        segment.arrays.replace_values(a, old, new, b)
        np.testing.assert_array_equal(a, [100, 101, 102, 103, 104, 105])
        np.testing.assert_array_equal(b, [0, 10, 2, 3, 40, 50])

        a = np.array([1e6, 1e12], dtype=np.uint64)
        b = a.copy()
        old = np.array([1e6], dtype=np.uint64)
        new = np.array([1], dtype=np.uint64)
        b = segment.arrays.replace_values(a, old, new, b)
        np.testing.assert_array_equal(a, [1e6, 1e12])
        np.testing.assert_array_equal(b, [1, 1e12])

        a = np.array([1e6, 1e12], dtype=np.uint64)
        old = np.array([1e12], dtype=np.uint64)
        new = np.array([2], dtype=np.uint64)
        b = segment.arrays.replace_values(a, old, new, b)
        np.testing.assert_array_equal(a, [1e6, 1e12])
        np.testing.assert_array_equal(b, [1, 2])

        a = np.array([0, 1e6, 1e12], dtype=np.uint64)
        old = np.array([1e6], dtype=np.uint64)
        new = np.array([1], dtype=np.uint64)
        b = segment.arrays.replace_values(a, old, new)

        np.testing.assert_array_equal(a, [0, 1e6, 1e12])
        np.testing.assert_array_equal(b, [0, 1, 1e12])

    def test_relabel(self):
        a = np.array([0, 1, 2, 3, 4, 5])
        b, n, bmap = segment.arrays.relabel(a, return_backwards_map=True)

        assert n == 5
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(bmap, [0, 1, 2, 3, 4, 5])

        a = np.array([0])
        b, n, bmap = segment.arrays.relabel(a, return_backwards_map=True)

        assert n == 0
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(bmap, [0])

        a = np.array([0])
        b, n = segment.arrays.relabel(a)

        assert n == 0
        np.testing.assert_array_equal(a, b)

        a = np.array([])
        b, n, bmap = segment.arrays.relabel(a, return_backwards_map=True)

        assert n == 0
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(bmap, [])

        a = np.array([])
        b, n = segment.arrays.relabel(a)

        assert n == 0
        np.testing.assert_array_equal(a, b)

        a = np.array([0, 100, 2, 1e8, 4, 5], dtype=np.uint64)
        b, n, bmap = segment.arrays.relabel(a, return_backwards_map=True)

        assert n == 5
        np.testing.assert_array_equal(a, [0, 100, 2, 1e8, 4, 5])
        np.testing.assert_array_equal(b, [0, 4, 1, 5, 2, 3])
        np.testing.assert_array_equal(bmap, [0, 2, 4, 5, 100, 1e8])

        a = np.array([0, 100, 2, 1e8, 4, 5], dtype=np.uint64)
        b, n = segment.arrays.relabel(a, inplace=True)

        assert n == 5
        np.testing.assert_array_equal(a, [0, 4, 1, 5, 2, 3])
        np.testing.assert_array_equal(b, [0, 4, 1, 5, 2, 3])
        np.testing.assert_array_equal(bmap, [0, 2, 4, 5, 100, 1e8])
