#include <vector>
#include <map>
#include <cstdint>
#include <boost/pending/disjoint_sets.hpp>
#include "find_components_impl.h"

void find_components_impl(
	size_t numNodes,
	size_t numEdges,
	const uint64_t* nodes, // 1D array of node IDs
	const uint64_t* edges, // 2D array of [(u,v)]
	uint64_t* components) {

	// disjoint sets datastructure to keep track of cluster merging
	std::vector<size_t> rank(numNodes);
	std::vector<uint64_t> parent(numNodes);
	std::map<uint64_t, std::size_t> node_to_set;
	boost::disjoint_sets<size_t*, uint64_t*> clusters(&rank[0], &parent[0]);

	for (size_t i = 0; i < numNodes; i++) {

		// initially, every node is in its own cluster
		node_to_set[nodes[i]] = i;
		clusters.make_set(i);
	}

	// merge edges
	for (size_t i = 0; i < numEdges; i++) {

		uint64_t u = edges[i*2];
		uint64_t v = edges[i*2 + 1];

		clusters.union_set(node_to_set[u], node_to_set[v]);
	}

	// label components array
	for (size_t i = 0; i < numNodes; i++) {

		components[i] = nodes[clusters.find_set(i)];
	}
}

