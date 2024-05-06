import numpy as np
from numpy.linalg import svd
import networkx as nx

from graph_utils import get_adj
from graph_visu import plot_graph


if __name__ == "__main__":
	G = nx.random_geometric_graph(5, 0.7)

	plot_graph(G)

	A = get_adj(G)
	print(A)
