from __future__ import absolute_import
from .impl import find_components
from .replace_values import replace_values
import daisy
import glob
import malis
import numpy as np
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

def relabel_connected_components(array_in, array_out, debug_array, block_size, num_workers):
    '''Relabel connected components in an array in parallel.

    Args:

        array_in (``daisy.Array``):

            The array to relabel.

        array_out (``daisy.Array``):

            The array to write to. Should initially be empty (i.e., all zeros).

        block_size (``daisy.Coordinate``):

            The size of the blocks to relabel in, in world units.

        num_workers (``int``):

            The number of workers to use.
    '''

    '''

    for debugging --> first connected components writes out to
    'array_out', second connected components writes out to 'debug_array'

    '''

    write_roi = daisy.Roi(
        (0,)*len(block_size),
        block_size)
    read_roi = write_roi.grow(array_in.voxel_size, array_in.voxel_size)
    total_roi = array_in.roi.grow(array_in.voxel_size, array_in.voxel_size)

    with tempfile.TemporaryDirectory() as tmpdir:

        daisy.run_blockwise(
            total_roi,
            read_roi,
            write_roi,
            process_function=lambda b: find_components_in_block(
                array_in,
                array_out,
                b,
                tmpdir),
            num_workers=num_workers,
            fit='shrink')

        # error either happens here after writing nodes/edges to npz

        nodes, edges = read_cross_block_merges(tmpdir)

    #or here?

    components = find_components(nodes, edges)

    logger.debug("Num nodes: %s", len(nodes))
    logger.debug("Num edges: %s", len(edges))
    logger.debug("Num components: %s", len(components))

    write_roi = daisy.Roi(
        (0,)*len(block_size),
        block_size)
    read_roi = write_roi
    total_roi = array_in.roi

    daisy.run_blockwise(
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda b: relabel_in_block(
            array_out,
            debug_array,
            nodes,
            components,
            b),
        num_workers=num_workers,
        fit='shrink')


def find_components_in_block(array_in, array_out, block, tmpdir):

    simple_neighborhood = malis.mknhood3d()

    labels = array_in.to_ndarray(block.read_roi, fill_value=0)

    affs = malis.seg_to_affgraph(labels, simple_neighborhood)

    components, _ = malis.connected_components_affgraph(
        affs,
        simple_neighborhood)

    num_voxels = (block.write_roi / array_in.voxel_size).size()

    components += block.block_id * num_voxels

    components[labels==0] = 0

    array_out[block.write_roi] = components[1:-1, 1:-1, 1:-1]

    neighbors = array_out.to_ndarray(roi=block.read_roi, fill_value=0)

    unique_pairs = []

    debug_ids = [
            8518254597,
            8490483715,
            8499740674,
            8555282463,
            8453455877,
            8444198920,
            8508997633,
            8555282461
            ]

    for d in range(3):

        slices_neg = tuple(
            slice(None) if dd != d else slice(0, 1)
            for dd in range(3)
        )
        slices_pos = tuple(
            slice(None) if dd != d else slice(-1, None)
            for dd in range(3)
        )

        pairs_neg = np.array([components[slices_neg].flatten(), neighbors[slices_neg].flatten()])
        pairs_neg = pairs_neg.transpose()

        # logger.debug('Pairs neg %s', pairs_neg.shape)

        pairs_pos = np.array([components[slices_pos].flatten(), neighbors[slices_pos].flatten()])
        pairs_pos = pairs_pos.transpose()

        # logger.debug('Pairs pos %s', pairs_pos.shape)

        unique_pairs.append(
            np.unique(
                np.concatenate([pairs_neg, pairs_pos]),
                axis=0))

    # Pretty sure the problem happens somewhere here, we have already written
    # the correct first connected components array out (array_out), we likely
    # break something when writing nodes to npz

    unique_pairs = np.concatenate(unique_pairs)
    zero_u = unique_pairs[:, 0] == 0
    zero_v = unique_pairs[:, 1] == 0
    non_zero_filter = np.logical_not(np.logical_or(zero_u, zero_v))

    edges = unique_pairs[non_zero_filter]
    nodes = np.unique(unique_pairs)

    for (u, v) in edges:
        if u in debug_ids or v in debug_ids:
            logger.debug("%d == %d", u, v)
            if u not in debug_ids or v not in debug_ids:
                logger.debug("%d != %d (wrong merge)", u, v)

    np.savez_compressed(
        os.path.join(tmpdir, 'block_%d.npz' % block.block_id),
        nodes=nodes,
        edges=edges)


def relabel_in_block(array, debug_array, old_values, new_values, block):

    # dont think anything bad happens here

    a = array.to_ndarray(block.write_roi)
    replace_values(a, old_values, new_values, inplace=True)
    debug_array[block.write_roi] = a

def read_cross_block_merges(tmpdir):

    block_files = glob.glob(os.path.join(tmpdir, 'block_*.npz'))

    nodes = []
    edges = []
    for block_file in block_files:
        b = np.load(block_file)
        nodes.append(b['nodes'])
        edges.append(b['edges'])

    return np.concatenate(nodes), np.concatenate(edges)
