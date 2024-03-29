import numpy as np
from scipy.spatial.distance import pdist, squareform


class NeighborhoodDeviationScore:
    def __init__(self, dist_matrix_: np.ndarray, n_reference: int, k_nearest_: int):
        """
        Class for computing NDS scores
        @param dist_matrix_: Distance matrix for the trajectories
        @param n_reference: Number of reference trajectories
        @param k_nearest_: Number of nearest neighbors to consider for each reference trajectory
        """
        self.dist_matrix = dist_matrix_
        # randomly choose reference trajectories
        self.ref_traj = np.random.choice(np.arange(len(dist_matrix_)), size=n_reference, replace=False)
        # compute nearest neighbors for each reference trajectory
        argsorted = np.argsort(dist_matrix_[self.ref_traj])
        self.nearest_ind = argsorted[:, :k_nearest_]
        # compute orders of nearest neighbors
        self.orders = np.array([np.arange(k_nearest_)] * n_reference)

    def compute_scores(self, embed, output_metric="euclidean", p=1):
        """
        Computes NDS scores for the given embedding
        @param embed: Embedding
        @param output_metric: Metric for computing distances in the embedding space
        @param p: Power for computing NDS
        @return: NDS
        """
        embed_dist_matrix = squareform(pdist(embed, metric=output_metric))
        embedding_orders = np.argsort(np.argsort(embed_dist_matrix[self.ref_traj]))
        k_nearest_orders = np.array([orders[ind] for orders, ind in zip(embedding_orders, self.nearest_ind)])
        return (np.abs(k_nearest_orders - self.orders) ** p).mean(axis=1) ** (1 / p) / len(embed)

    def nds(self, embed, output_metric="euclidean", p=1):
        """
        Computes mean and standard error of NDS scores for the given embedding
        @param embed: Embedding
        @param output_metric: Metric for computing distances in the embedding space
        @param p: Power for computing NDS
        @return: Returns mean and standard error of NDS scores
        """
        scores = self.compute_scores(embed, output_metric, p)
        return scores.mean(), scores.std() / np.sqrt(len(scores))
