import numpy as np
from sklearn.neighbors import KDTree


def ranks_metric(test_traj, transformed_traj, loss_fn=lambda x: np.exp2(x) - 1):
    tree = KDTree(test_traj)
    neighbors = tree.query(test_traj, k=test_traj.shape[0], return_distance=False)
    ranking = np.argsort(neighbors, axis=1)
    loss = 1

    closest_idxs = tree.query(transformed_traj, k=1, return_distance=False)[:, 0]
    print(closest_idxs)
    for i, closest_idx in enumerate(closest_idxs):
        rank = ranking[i][closest_idx]
        loss += loss_fn(rank)

    return np.log2(loss)


def mse_neighborhood_metric(test_traj, transformed_traj, n_neighbors=10):
    tree = KDTree(test_traj)
    neighbors = tree.query(test_traj, k=n_neighbors, return_distance=False)
    # return mean_squared_error(test_traj[neighbors], transformed_traj[neighbors])
    mse = ((test_traj[neighbors] - transformed_traj[neighbors])**2).mean(axis=-1).mean(axis=-1) # todo
    return np.min(mse)
