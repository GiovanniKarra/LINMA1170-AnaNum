
from sklearn.datasets import make_circles
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt

# generate your data
X, labels = make_circles(n_samples=500, noise=0.1, factor=.2)

# plot your data
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# train and predict
s_cluster = SpectralClustering(n_clusters=2, eigen_solver='arpack',
                               affinity="nearest_neighbors").fit_predict(X)

# plot clustered data
plt.scatter(X[:, 0], X[:, 1], c=s_cluster)
plt.show()