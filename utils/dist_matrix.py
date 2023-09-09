import numpy as np
import wasserstein.wasserstein as ws
from sklearn.neighbors import KDTree
from tqdm import tqdm


# def sinkhorn_dist_matrix(data: np.ndarray) -> np.ndarray:
# return sinkhorn_divergence(pointcloud.PointCloud, x=data[i], y=data[j]).divergence # TODO: debias

def gen_dist_matrix(X: np.ndarray, use_sinkhorn=False, beta=2, name=None, verbosity=1) -> np.ndarray:
    """
    Generates distance matrix by calculating Wasserstein distance
    @param X: Trajectories data
    @param use_sinkhorn:
    @return: Distance matrix
    """
    weighted_data = np.array([np.array([np.array([1] + list(b)) for b in a]) for a in X])
    pw_emd = ws.PairwiseEMD(beta=beta, verbose=verbosity)
    pw_emd(weighted_data)
    dist_matrix = pw_emd.emds()
    dist_matrix[dist_matrix < 0] = 0
    if not (name is None):
        np.savez("dist_matrix_" + name + ".npz", dist_matrix)
    return dist_matrix ** (1 / beta)


def uniform_on_interval(a: np.ndarray, b: np.ndarray, N_points: int) -> list:
    """
    Creates evenly spaced points on the interval
    @params a, b: bounds of the interval
    @param N_points: the number of points to generate
    @return: generated points
    """
    if N_points == 0:
        return [a]
    return [x * b + (1 - x) * a for x in np.linspace(0, 1, N_points)]


def uniform_points_on_traj(traj: np.ndarray, N_points: int) -> np.ndarray:
    """
    Creates evenly spaced points on the trajectory
    @param traj: trajectory
    @param N_traj: number of points to generate
    @return: generated points
    """
    interval_lengths = np.sqrt(((traj[1:] - traj[:-1]) ** 2).sum(axis=1))
    n_points_on_intervals = (interval_lengths / interval_lengths.sum() * N_points).astype(int)
    points = np.concatenate([uniform_on_interval(traj[i], traj[i + 1], n_points_on_intervals[i]) for i in range(len(traj) - 1)])
    return points


def query(kdtree, points):
    return kdtree.query(points)[0].sum()


def gen_unbiased_dist_matrix(X: np.ndarray, N_points: int=10000) -> np.ndarray:
    """
    Generates distance matrix by generating a lot of uniform points on trajectories and finding the nearest points
    @param X: Trajectories data
    @param N_points: number of points to generate on each trajectory for high precision
    @return: Distance matrix
    """
    generated_points = [uniform_points_on_traj(trajectory, N_points) for trajectory in X]
    kdtrees = [KDTree(points) for points in generated_points]
    
    res = np.zeros((len(X), len(X)))
    for i in tqdm(range(len(X))):
        for j in range(len(X)):
            if j < i:
                res[i][j] = res[j][i]
            else:
                res[i][j] = (query(kdtrees[i], generated_points[j]) + query(kdtrees[j], generated_points[i])) / 2 / N_points
                
    return res
