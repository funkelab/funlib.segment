import daisy
from funlib import segment

if __name__ == "__main__":

    graph = daisy.Graph()

    graph.add_node(3)
    graph.add_node(4)
    graph.add_node(100)
    graph.add_edge(3, 4, score=0.5)

    lut = segment.graphs.find_connected_components(
        graph,
        edge_score_attribute='score',
        edge_score_relation='<=',
        edge_score_threshold=0.5)

    print(lut)
