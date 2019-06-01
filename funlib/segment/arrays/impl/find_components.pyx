import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t


cdef extern from "find_components_impl.h":
    void find_components_impl(
        size_t numNodes,
        size_t numEdges,
        const uint64_t* nodes,
        const uint64_t* edges,
        uint64_t* components);


def find_components(
    np.ndarray[uint64_t, ndim=1] nodes,
    np.ndarray[uint64_t, ndim=2] edges):
    '''Find connected components.
    '''

    cdef size_t num_nodes = nodes.shape[0]
    cdef size_t num_edges = edges.shape[0]

    assert edges.shape[1] == 2, "edges not given as rows of [u, v]"

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not nodes.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous nodes arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        nodes = np.ascontiguousarray(nodes)
    if not edges.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous edges arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        edges = np.ascontiguousarray(edges)

    # prepare output arrays
    cdef np.ndarray[uint64_t, ndim=1] components = np.zeros(
            (num_nodes,),
            dtype=np.uint64)

    find_components_impl(
        num_nodes,
        num_edges,
        &nodes[0],
        &edges[0, 0],
        &components[0])

    return components
