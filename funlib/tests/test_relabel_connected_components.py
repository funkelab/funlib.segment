from funlib import segment
import daisy
import numpy as np
import os
import tempfile
import unittest

daisy.scheduler._NO_SPAWN_STATUS_THREAD = True


class TestRelabelConnectedComponents(unittest.TestCase):

    def test_relabel_connected_components(self):

        roi = daisy.Roi(
            (0, 0, 0),
            (100, 100, 100))

        block_size = (25, 25, 25)

        with tempfile.TemporaryDirectory() as tmpdir:

            array_in = daisy.prepare_ds(
                os.path.join(tmpdir, 'array_in.zarr'),
                'volumes/in',
                roi,
                voxel_size=(1, 1, 1),
                dtype=np.uint64)

            in_data = np.zeros((100, 100, 100), dtype=np.uint64)
            in_data[20] = 1
            in_data[40] = 1
            in_data[60] = 2

            array_in[roi] = in_data

            array_out = daisy.prepare_ds(
                os.path.join(tmpdir, 'array_out.zarr'),
                'volumes/out',
                roi,
                voxel_size=(1, 1, 1),
                write_size=block_size,
                dtype=np.uint64)

            segment.arrays.relabel_connected_components(
                array_in,
                array_out,
                block_size=block_size,
                num_workers=10)

            out_data = array_out.to_ndarray(roi)

        assert not np.testing.assert_array_equal(out_data[20], out_data[40])
        assert not np.testing.assert_array_equal(out_data[40], out_data[60])
        assert out_data[0:20].sum() == 0
        assert out_data[21:40].sum() == 0
        assert out_data[41:60].sum() == 0
        assert out_data[61:].sum() == 0
        assert len(np.unique(out_data[20])) == 1
        assert len(np.unique(out_data[40])) == 1
        assert len(np.unique(out_data[60])) == 1
