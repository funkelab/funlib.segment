from .segment_blockwise import segment_blockwise
import daisy
import skimage.measure


def label_connected_components(array_in, roi):
    labels = array_in.to_ndarray(roi, fill_value=0)
    components = skimage.measure.label(labels, connectivity=1).astype(labels.dtype)
    components[labels == 0] = 0

    return components


def relabel_connected_components(array_in, array_out, block_size, num_workers):
    """Relabel connected components in an array in parallel.

    Args:

        array_in (``daisy.Array``):

            The array to relabel.

        array_out (``daisy.Array``):

            The array to write to. Should initially be empty (i.e., all zeros).

        block_size (``daisy.Coordinate``):

            The size of the blocks to relabel in, in world units.

        num_workers (``int``):

            The number of workers to use.
    """

    block_size = daisy.Coordinate(block_size)

    segment_blockwise(
        array_in,
        array_out,
        block_size=block_size,
        context=array_in.voxel_size,
        num_workers=num_workers,
        segment_function=label_connected_components,
    )
