from funlib import segment
from funlib.persistence.graphs.graph import Graph
import unittest


class TestGraphCc(unittest.TestCase):
    def test_graph_cc(self):
        graph = Graph()

        graph.add_node(3)
        graph.add_node(4)
        graph.add_node(100)
        graph.add_edge(3, 4, score=0.5)

        lut = segment.graphs.find_connected_components(graph)

        assert lut[3] == lut[4]
        assert lut[3] != lut[100]

        lut = segment.graphs.find_connected_components(
            graph,
            edge_score_attribute="score",
            edge_score_relation="<=",
            edge_score_threshold=0.5,
        )

        assert lut[3] == lut[4]
        assert lut[3] != lut[100]

        lut = segment.graphs.find_connected_components(
            graph,
            edge_score_attribute="score",
            edge_score_relation="<=",
            edge_score_threshold=0.4,
        )

        assert lut[3] != lut[4]
        assert lut[3] != lut[100]
        assert lut[4] != lut[100]

    def test_graph_cc_writeback(self):
        graph = Graph()

        graph.add_node(3)
        graph.add_node(4)
        graph.add_node(100)
        graph.add_edge(3, 4, score=0.5)

        lut = segment.graphs.find_connected_components(
            graph,
            edge_score_attribute="score",
            edge_score_relation="<=",
            edge_score_threshold=0.5,
            node_component_attribute="component",
        )

        assert lut[3] == lut[4]
        assert lut[3] != lut[100]
        assert graph.nodes[3]["component"] == 1
        assert graph.nodes[4]["component"] == 1
        assert graph.nodes[100]["component"] == 2

    def test_api_fail(self):
        graph = Graph()

        graph.add_node(3)
        graph.add_node(4)
        graph.add_node(100)
        graph.add_edge(3, 4, score=0.5)

        with self.assertRaises(RuntimeError):
            segment.graphs.find_connected_components(graph, return_lut=False)

        with self.assertRaises(RuntimeError):
            segment.graphs.find_connected_components(
                graph, edge_score_attribute="score", edge_score_relation=">"
            )
