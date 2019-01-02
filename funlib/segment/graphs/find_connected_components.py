from __future__ import absolute_import
from .impl import connected_components
import numpy as np


def find_connected_components(
        graph,
        node_component_attribute=None,
        edge_score_attribute=None,
        edge_score_relation='<=',
        edge_score_threshold=None,
        return_lut=True):
    '''Label connected components in the given graph.

    By default, connectivitiy is induced by the existence of edges.

    If ``edge_attribute`` is not ``None``, the given edge attribute and
    relation will be used to infer connectivity: if the edge attribute is
    smaller (or larger) than the given ``threshold``, the incident nodes are
    considered connected. E.g., if the edge attritute is ``score``, the
    relation ``<=`` and the threshold ``0.5``, every edge with
    ``graph.edges[e]["score"] <= 0.5`` will be considered for connectivity.

    Arguments:

        graph (``networkx``-compatible graph):

            The graph to find connected components in.

        node_component_attribute (``string``, optional):

            If given, the component number will be written to that node
            attribute.

        edge_score_attribute (``string``, optional):

            Which edge attribute to use to determine connectivity. If not
            given, connectivity is induced by the existence of edges.

        edge_score_relation (``string``, optional):

            Any of ``<``, ``<=``, ``>``, ``>=``.

        edge_score_threshold (``float``, optional):

            Which threshold to use on ``edge_score_attribute`` to determine
            connectivity.

        return_lut (``bool``, optional):

            Return a look-up table from nodes to components.
    '''

    if node_component_attribute is None and not return_lut:
        raise RuntimeError(
            "Either 'node_component_attribute' or 'return_lut' has to be "
            "given.")

    # convert the graph into a continuous memory representation for the C++
    # backend

    if edge_score_attribute is not None and edge_score_relation != '<=':
        raise RuntimeError("Other relations than '<=' not yet implemented.")

    nodes = np.array(graph.nodes, dtype=np.uint64)
    edges = np.array(list([e[0], e[1]] for e in graph.edges), dtype=np.uint64)

    if edge_score_attribute is not None:
        scores = np.array(
            list(e[2] for e in graph.edges(data=edge_score_attribute)),
            dtype=np.float32)
    else:
        scores = np.ones(len(graph.edges), dtype=np.float32)
        edge_score_threshold = 1

    # find connected components
    components = connected_components(
        nodes,
        edges,
        scores,
        edge_score_threshold)

    if node_component_attribute is not None:

        # write back node attributes
        for node, component in zip(nodes, components):
            graph.nodes[node][node_component_attribute] = component

    if return_lut:

        return {
            node: component
            for node, component in zip(nodes, components)
        }
