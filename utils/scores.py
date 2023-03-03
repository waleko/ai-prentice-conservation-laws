import numpy as np
from sklearn.manifold import Isomap
from scipy.spatial import distance_matrix

def order_scores(dist_matrix, n_dims_arr=np.arange(1, 5), n_trajectories=20):
    random_trajectories = np.random.choice(np.arange(dist_matrix.shape[0]), replace=False, size=n_trajectories)
    initial_orders = np.argsort(np.argsort(dist_matrix[random_trajectories]))
    scores_lst = []
    for n_dims in n_dims_arr:
        embedder = Isomap(n_components=n_dims, metric="precomputed")
        embedding = embedder.fit_transform(dist_matrix)
        embedding_orders = np.argsort(np.argsort(distance_matrix(embedding[random_trajectories], embedding)))
        score = np.abs(initial_orders - embedding_orders).mean() / dist_matrix.shape[0]
        scores_lst.append(score)
    return scores_lst
