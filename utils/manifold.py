import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import scipy

def spectral_embedding(W: np.ndarray, n_neighbors: int = 20, alpha: float = 1.0, cutoff: float = 0.6, skip_plot=False):
    N = W.shape[0]
    nn_max = np.mean(np.sort(W)[:,n_neighbors+1])
    eps = 2 * nn_max**2
    # Calculate affinity matrix and get embedding
    K = np.exp(-(W ** 2) / eps) # kernel matrix
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
    mean_shift = np.mean(diag / sqrt_norm**2)
    np.fill_diagonal(K, 1 - mean_shift)

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(-K, 20, which="LM", sigma=1.0, v0=None)
    # print(eigenvalues, eigenvectors)

    eigenvectors = eigenvectors[-2::-1]
    eigenvalues = eigenvalues[-2::-1] # ignore eigenvalue = 1.0

    # renormalize
    eigenvectors = (
        np.sqrt(eigenvectors.shape[0]) * eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    )

    scores, embed_list = heuristic_score(eigenvalues, eigenvectors, cutoff, 5)

    # Plot components
    comp_color = [ "black" if x >= cutoff else "grey" for x in scores]
    if not skip_plot:
        plt.bar(np.arange(1, len(scores) + 1), scores, color=comp_color)
        plt.axhline(y=cutoff, color = 'blue', linestyle = '--')
        plt.xlabel("Component")
        plt.ylabel("Score")
        plt.title(f"Components with cutoff {cutoff}, eps={eps}")
        plt.show()
    return scores, eigenvectors, embed_list

def heuristic_score(evals, evecs, cutoff, n_neighbors):
    n_components = evals.shape[0]
    
    # length scale using log (see Appendix B)
    weights = np.empty_like(evals)
    weights[evals <= -1] = 0
    weights[evals > -1] = np.sqrt(
        np.log(1 + evals[0]) / np.log(1 + evals[evals > -1])
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
