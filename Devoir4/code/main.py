import numpy as np
from numpy.linalg import svd
import networkx as nx
import matplotlib.pyplot as plt

from graph_utils import get_adj
from graph_visu import plot_graph_plotly


if __name__ == "__main__":
	np.set_printoptions(precision=2, suppress=True)
	G = nx.random_graphs.gnp_random_graph(5, 0.5)

	# plot_graph(G)
	fig = plt.figure()
	ax = fig.subplots()

	pos = nx.spring_layout(G)
	nx.draw(G, pos=pos, ax=ax)

	A = get_adj(G)

	U, S, VH = svd(A)

	print(VH.T)

	for i, (x, y) in pos.items():
		ax.text(x-0.02, y-0.02, str(i), fontsize=14, color="white")

	plt.show()