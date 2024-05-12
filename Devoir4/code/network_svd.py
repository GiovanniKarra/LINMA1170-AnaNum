from IPython.display import SVG
import numpy as np
from sknetwork.utils import get_degrees
from sknetwork.data import from_edge_list, karate_club, painters, movie_actor, star_wars, miserables
from sknetwork.embedding import SVD
from sknetwork.visualization import visualize_graph, visualize_bigraph
from scipy import sparse


# Graph
graph = karate_club(metadata=True)
adjacency = graph.adjacency
labels = graph.labels
svd = SVD(2)
embedding = svd.fit_transform(adjacency)
#Pour sauvegarder une image et la visualiser, ajouter l'argument filename='...' à la fonction visualize_graph
image = visualize_graph(adjacency, embedding, labels=labels)


# Graph - Pas encore de résultat concluant
graph = karate_club(metadata=True)
adjacency = graph.adjacency
degree_vector = get_degrees(adjacency)
degree_matrix = np.diag(degree_vector)
laplacian = sparse.csr_matrix(degree_matrix - adjacency)
embedding = svd.fit_transform(laplacian)
image = visualize_graph(adjacency, embedding, labels=labels)



#Directed Graph
graph = painters(metadata=True)
adjacency = graph.adjacency
names = graph.names
svd = SVD(2)
embedding = svd.fit_transform(adjacency)
image = visualize_graph(adjacency, embedding, names=names)


#Bipartite graph
graph = star_wars(metadata=True)
biadjacency = graph.biadjacency
names_row = graph.names_row
names_col = graph.names_col
svd = SVD(2, normalized=False)
svd.fit(biadjacency)
SVD(n_components=2, regularization=None, factor_singular=0.0, normalized=False)
embedding_row = svd.embedding_row_
embedding_col = svd.embedding_col_
image = visualize_bigraph(biadjacency, names_row, names_col,
                    position_row=embedding_row, position_col=embedding_col,
                    color_row='blue', color_col='red', scale=1.5)


#Bipartite graph
graph = movie_actor(metadata=True)
biadjacency = graph.biadjacency
names_row = graph.names_row
names_col = graph.names_col
svd = SVD(2, normalized=False)
svd.fit(biadjacency)
SVD(n_components=2, regularization=None, factor_singular=0.0, normalized=False)
embedding_row = svd.embedding_row_
embedding_col = svd.embedding_col_
image = visualize_bigraph(biadjacency, names_row, names_col,
                    position_row=embedding_row, position_col=embedding_col,
                    color_row='blue', color_col='red', scale=1.5)

#self-build graph
relations = np.array([('Alice','Bob'), ('Alice','Cecilia'), ('Bob','Cecilia'), ('Cecilia','David'), ('David','Alice')
                      , ('David', 'Estelle'), ('Estelle', 'Floriane'), ('Estelle', 'Giobane'), ('Floriane', 'Giobane')])
graph = from_edge_list(relations)
adjacency = graph.adjacency
names = graph.names
svd = SVD(2)
embedding = svd.fit_transform(adjacency)
#self-build graph after svd
image = visualize_graph(adjacency, embedding, names=names)

#self build graph before svd (complete)
image_2 = visualize_graph(adjacency, names=names)