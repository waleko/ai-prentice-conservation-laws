import numpy as np
import wasserstein.wasserstein as ws


def gen_dist_matrix(X: np.ndarray, beta=2, name=None) -> np.ndarray:
    """
    Generates distance matrix by calculating Wasserstein distance
    @param X: Trajectories data
    @param name: Suffix for saving the distance matrix to a file
    @param beta: Wasserstein distance parameter (default: 2)
    @return: Distance matrix
    """
    weighted_data = np.array([np.array([np.array([1] + list(b)) for b in a]) for a in X])
    pw_emd = ws.PairwiseEMD(beta=beta)
    pw_emd(weighted_data)
    dist_matrix = pw_emd.emds()
    dist_matrix[dist_matrix < 0] = 0
    if not (name is None):
        np.savez("dist_matrix_" + name + ".npz", dist_matrix)
    return dist_matrix ** (1 / beta)

