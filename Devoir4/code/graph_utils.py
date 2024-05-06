import numpy as np
import networkx as nx


def get_adj(G: nx.Graph):
	n = G.number_of_nodes()

	A = np.zeros((n, n), dtype=int)

	for node, adj in G.adjacency():
		for neigh in adj.keys():
			A[node][neigh] = 1

	return A