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

    def order_score(self, embed, output_metric="euclidean", p=1):
        embed_dist_matrix = squareform(pdist(embed, metric=output_metric))
        embedding_orders = np.argsort(np.argsort(embed_dist_matrix[self.ref_traj]))
        k_nearest_orders = np.array([orders[ind] for orders, ind in zip(embedding_orders, self.nearest_ind)])
        return (np.abs(k_nearest_orders - self.orders) ** p).mean() ** (1 / p) / len(embed)


def order_scores(dist_matrix, UMAP_kwargs={"n_epochs": 2000, "n_neighbors": 80}, n_dims_arr=np.arange(1, 7), n_reference=20, k_nearest=40, check_periodic_1d=True):
    Score = OrderScore(dist_matrix, n_reference, k_nearest)
    scores_arr = []
    if check_periodic_1d:
        embed = UMAP(metric="precomputed", n_components=n_dims, output_metric=circle_metric, **UMAP_kwargs).fit_transform(dist_matrix)
        scores_arr.append(Score.order_score(embed, output_metric=circle_metric_without_grad))
    for n_dims in tqdm(n_dims_arr):
        embed = UMAP(metric="precomputed", n_components=n_dims, **UMAP_kwargs).fit_transform(dist_matrix)
        scores_arr.append(Score.order_score(embed))
    return scores_arr


def order_scores_different_n_ref(dist_matrix, n_reference_arr, n_repeats=5000, UMAP_kwargs={"n_epochs": 2000, "n_neighbors": 80}, n_dims_arr=np.arange(1, 7), k_nearest=40):
    Score_arr = [[OrderScore(dist_matrix, n_reference, k_nearest) for _ in range(n_repeats)] for n_reference in n_reference_arr]
    scores_arr = []
    for n_dims in tqdm(n_dims_arr):
        embed = UMAP(metric="precomputed", n_components=n_dims, **UMAP_kwargs).fit_transform(dist_matrix)
        scores_arr.append([[Score.order_score(embed) for Score in row] for row in Score_arr])
    return scores_arr
