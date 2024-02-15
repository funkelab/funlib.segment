from funlib import segment
import numpy as np
import unittest


class TestFindComponents(unittest.TestCase):
    def test_find_components(self):
        nodes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint64)
        edges = np.array(
            [[0, 1], [2, 3], [3, 7], [3, 7], [3, 7], [3, 7], [3, 7], [6, 2]],
            dtype=np.uint64,
        )

        components = segment.arrays.impl.find_components(nodes, edges)

        self.assertEqual(components[0], components[1])
        self.assertEqual(components[2], components[3])
        self.assertEqual(components[2], components[6])
        self.assertEqual(components[2], components[7])
        self.assertEqual(components[3], components[7])
        self.assertNotEqual(components[0], components[2])
        self.assertNotEqual(components[1], components[2])

    def test_find_components_large_ids(self):
        nodes = np.array([0, 1, 2, np.uint64(-1), 1000], dtype=np.uint64)
        edges = np.array([[0, 1], [2, 1000], [np.uint64(-1), 1000]], dtype=np.uint64)

        components = segment.arrays.impl.find_components(nodes, edges)

        self.assertEqual(components[0], components[1])
        self.assertEqual(components[2], components[3])
        self.assertEqual(components[2], components[4])
        self.assertEqual(components[3], components[4])
        self.assertNotEqual(components[0], components[2])
        self.assertNotEqual(components[1], components[2])
        self.assertNotEqual(components[1], components[4])

        for c in components:
            self.assertTrue(c in nodes)
