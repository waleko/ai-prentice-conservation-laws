import numpy as np
from umap import UMAP
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from .metrics import circle_metric, circle_metric_without_grad
    

class OrderScore:
    def __init__(self, dist_matrix_: np.ndarray, n_reference: int, k_nearest_: int):
        self.dist_matrix = dist_matrix_
        self.ref_traj = np.random.choice(np.arange(len(dist_matrix_)), size=n_reference, replace=False)
        argsorted = np.argsort(dist_matrix_[self.ref_traj])
        self.nearest_ind = argsorted[:, :k_nearest_]
        self.orders = np.array([np.arange(k_nearest_)] * n_reference)
    
    def compute_scores(self, embed, output_metric="euclidean", p=1):
        embed_dist_matrix = squareform(pdist(embed, metric=output_metric))
        embedding_orders = np.argsort(np.argsort(embed_dist_matrix[self.ref_traj]))
        k_nearest_orders = np.array([orders[ind] for orders, ind in zip(embedding_orders, self.nearest_ind)])
        return (np.abs(k_nearest_orders - self.orders) ** p).mean(axis=1) ** (1 / p) / len(embed)
    
    def order_score(self, embed, output_metric="euclidean", p=1):
        scores = self.compute_scores(embed, output_metric, p)
        return scores.mean(), scores.std() / np.sqrt(len(scores))
