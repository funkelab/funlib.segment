from funlib import segment

from funlib.persistence.arrays import Array, prepare_ds
import daisy
import numpy as np
import os
import tempfile
import unittest

daisy.scheduler._NO_SPAWN_STATUS_THREAD = True


class TestRelabelConnectedComponents(unittest.TestCase):
    def test_minimal(self):
        labels = np.array([[[1, 1, 1, 2, 2, 3, 2, 2, 1, 140, 140, 0]]], dtype=np.uint64)

        roi = daisy.Roi((0, 0, 0), labels.shape)
        voxel_size = (1, 1, 1)

        block_size = (1, 1, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            a = Array(labels, roi=roi, voxel_size=voxel_size)
            b = prepare_ds(
                os.path.join(tmpdir, "array_out.zarr"),
                "/volumes/b",
                total_roi=roi,
                voxel_size=voxel_size,
                write_size=block_size,
                dtype=np.uint64,
            )

            b.data[:] = 0

            segment.arrays.relabel_connected_components(a, b, block_size, 1)

            b = b.data[:].flatten()

            self.assertTrue(b[0] == b[1] == b[2])
            self.assertTrue(b[3] == b[4])
            self.assertTrue(b[6] == b[7])
            self.assertTrue(b[9] == b[10])
            self.assertTrue(b[2] != b[3])
            self.assertTrue(b[4] != b[5])
            self.assertTrue(b[5] != b[6])
            self.assertTrue(b[7] != b[8])
            self.assertTrue(b[8] != b[9])
            self.assertTrue(b[10] != b[11])

    def test_relabel_connected_components(self):
        roi = daisy.Roi((0, 0, 0), (100, 100, 100))

        block_size = (25, 25, 25)

        with tempfile.TemporaryDirectory() as tmpdir:
            array_in = prepare_ds(
                os.path.join(tmpdir, "array_in.zarr"),
                "volumes/in",
                roi,
                voxel_size=(1, 1, 1),
                dtype=np.uint64,
            )

            in_data = np.zeros((100, 100, 100), dtype=np.uint64)
            in_data[20] = 1
            in_data[40] = 1
            in_data[60] = 2

            array_in[roi] = in_data

            array_out = prepare_ds(
                os.path.join(tmpdir, "array_out.zarr"),
                "volumes/out",
                roi,
                voxel_size=(1, 1, 1),
                write_size=block_size,
                dtype=np.uint64,
            )

            segment.arrays.relabel_connected_components(
                array_in, array_out, block_size=block_size, num_workers=10
            )

            out_data = array_out.to_ndarray(roi)

        np.testing.assert_array_equal(out_data[20] == out_data[40], False)
        np.testing.assert_array_equal(out_data[40] == out_data[60], False)
        self.assertEqual(out_data[0:20].sum(), 0)
        self.assertEqual(out_data[21:40].sum(), 0)
        self.assertEqual(out_data[41:60].sum(), 0)
        self.assertEqual(out_data[61:].sum(), 0)
        self.assertEqual(len(np.unique(out_data[20])), 1)
        self.assertEqual(len(np.unique(out_data[40])), 1)
        self.assertEqual(len(np.unique(out_data[60])), 1)
