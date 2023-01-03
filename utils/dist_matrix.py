import numpy as np
import wasserstein.wasserstein as ws


# def sinkhorn_dist_matrix(data: np.ndarray) -> np.ndarray:
# return sinkhorn_divergence(pointcloud.PointCloud, x=data[i], y=data[j]).divergence # TODO: debias

def gen_dist_matrix(X: np.ndarray, use_sinkhorn=False) -> np.ndarray:
    """
    Generates distance matrix by calculating Wasserstein distance
    @param X: Trajectories data
    @param use_sinkhorn:
    @return: Distance matrix
    """
    weighted_data = np.array([np.array([np.array([1] + list(b)) for b in a]) for a in X])
    pw_emd = ws.PairwiseEMD()
    pw_emd(weighted_data)
    return pw_emd.emds()
