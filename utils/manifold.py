import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import scipy


def spectral_embedding(W: np.ndarray, n_neighbors: int = 20, alpha: float = 1.0, cutoff: float = 0.6, do_plot=True,
                       n_components: int = 20):
    """
    Calculates evals, embedding by distance matrix using diffusion maps
    @param W: distance matrix
    @param n_neighbors: number of neighbors for calculating epsilon
    @param alpha:
    @param cutoff: heuristic score threshold
    @param do_plot: whether to plot components heuristic score
    @param n_components: number of components for KNN in heuristic score
    @return: evals, embedding and embed list
    """
    # Calculate epsilon
    nn_max = np.mean(np.sort(W)[:, n_neighbors + 1])
    eps = 2 * nn_max ** 2

    # Kernel matrix
    K = np.exp(-(W ** 2) / eps)
    alpha_norm = K.sum(axis=1) ** alpha
    K /= alpha_norm
    K /= alpha_norm[:, None]

    diag = K.diagonal().copy()
    np.fill_diagonal(K, 0)

    sqrt_norm = np.sqrt(K.sum(axis=1))
    K /= sqrt_norm
    K /= sqrt_norm[:, None]

    # Graph laplacian GL = 1 - L
    K *= -1
    mean_shift = np.mean(diag / sqrt_norm ** 2)
    np.fill_diagonal(K, 1 - mean_shift)

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(-K, k=n_components + 1, which="LM", sigma=1.0, v0=None)

    eigenvectors = eigenvectors[:, n_components - 1 :: -1] / sqrt_norm[:, None]
    eigenvalues = eigenvalues[n_components - 1 :: -1]

    # renormalize
    eigenvectors = (
            np.sqrt(eigenvectors.shape[0]) * eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    )

    scores, embed_list = heuristic_score(eigenvalues, eigenvectors, cutoff)

    # Plot components
    comp_color = ["black" if x >= cutoff else "grey" for x in scores]
    if do_plot:
        plt.bar(np.arange(1, len(scores) + 1), scores, color=comp_color)
        plt.axhline(y=cutoff, color='blue', linestyle='--')
        plt.xlabel("Component")
        plt.ylabel("Score")
        plt.title(f"Components with cutoff {cutoff}, eps={eps}")
        plt.show()
    return scores, eigenvectors, embed_list


def heuristic_score(evals: np.ndarray, evecs: np.ndarray, cutoff: float, n_neighbors: int = 5):
    """
    Calculates heuristic score described in Appendix B of the paper
    @param evals: eigenvalues of embedding
    @param evecs: eigenvectors of embedding
    @param cutoff: threshold of heuristic score
    @param n_neighbors:
    @return: scores and embed list
    """
    n_components = evals.shape[0]

    # length scale using log (see Appendix B)
    relevant_idx = evals > -1
    weights = np.empty_like(evals)
    weights[~relevant_idx] = 0
    weights[relevant_idx] = np.sqrt(
        np.log(1 + evals[0]) / np.log(1 + evals[relevant_idx])
    )

    n_trajectories = evecs.shape[0]
    n_off_diag_entries = n_trajectories * (n_trajectories - 1)
    mean_pairwise_dist_0 = np.sum(
        np.abs(evecs[:, None, 0] - evecs[None, :, 0]) / n_off_diag_entries
    )
    embed_list = [0]
    scores = [mean_pairwise_dist_0 * weights[0]]

    nearest_neighbors = neighbors.KNeighborsTransformer(n_neighbors=n_neighbors)
    for i in range(1, n_components):
        current_embedding = evecs[:, embed_list]
        candidate_vec = evecs[:, i]
        nearest_neighbors.fit(current_embedding)
        nn_ind = nearest_neighbors.kneighbors(return_distance=False)

        score = np.mean(np.abs(candidate_vec[:, None] - candidate_vec[nn_ind]))
        score *= weights[i]
        scores.append(score)
        if score >= cutoff:
            embed_list.append(i)
    return scores, embed_list


def batch_spectral_embedding(Ws: np.ndarray, n_neighbors: int = 20, alpha: float = 1.0, cutoff: float = 0.6, do_plot=True,
                             n_components: int = 20):
    res = []
    for W in Ws:
        res.append(spectral_embedding(W, n_neighbors, alpha, cutoff, False, n_components))
    return res