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


def order_scores(dist_matrix, UMAP_kwargs={"n_neighbors": 80}, n_dims_arr=np.arange(1, 7), n_reference=20, k_nearest=40, check_periodic_1d=True):
    Score = OrderScore(dist_matrix, n_reference, k_nearest)
    scores_arr = []
    if check_periodic_1d:
        embed = UMAP(metric="precomputed", n_components=1, output_metric=circle_metric, **UMAP_kwargs).fit_transform(dist_matrix)
        scores_arr.append(Score.order_score(embed, output_metric=circle_metric_without_grad))
    for n_dims in tqdm(n_dims_arr):
        embed = UMAP(metric="precomputed", n_components=n_dims, **UMAP_kwargs).fit_transform(dist_matrix)
        scores_arr.append(Score.order_score(embed))
    return scores_arr
