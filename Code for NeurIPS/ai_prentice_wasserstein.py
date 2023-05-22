import warnings
from functools import partial
from typing import Union

import numpy as np
import umap
from sklearn.preprocessing import StandardScaler

from utils.dist_matrix import gen_dist_matrix
from utils.early_stopping import get_stop_point
from utils.metrics import circle_metric, normalize_angle, circle_metric_without_grad
from utils.scores import NeighborhoodDeviationScore


class DimensionalityPrentice:
    def __init__(self, normalize: bool = True, beta: float = 2.0, n_neighbors: Union[int, float] = 0.5,
                 n_reference: Union[int, float] = 40, n_nearest: Union[int, float] = 0.25, max_dim: int = 6,
                 n_epochs_1d: int = 20000, threshold: float = 0.01, verbosity: int = 1):
        """
        Class for computing the dimensionality of a dataset using the method from

        @param normalize: Whether to normalize the data
        @param beta: Beta parameter for the Wasserstein distance
        @param n_neighbors: Number of neighbors for the UMAP embedding
        @param n_reference: Number of reference trajectories for the NDS score
        @param n_nearest: Number of nearest neighbors for the NDS score
        @param max_dim: Maximum dimensionality to consider
        @param n_epochs_1d: Number of epochs for the 1D UMAP embedding
        @param threshold: Threshold for stopping
        @param verbosity: Verbosity level
        """
        self.normalize = normalize
        self.beta = beta
        self.n_neighbors = n_neighbors
        self.n_reference = n_reference
        self.n_nearest = n_nearest
        self.max_dim = max_dim
        self.n_epochs_1d = n_epochs_1d
        self.threshold = threshold
        self.verbosity = verbosity

    def _convert_to_int(self, val, N):
        if type(val) == float:
            return int(val * N)
        return val

    def fit(self, data: np.ndarray, ws_distance_matrix: np.ndarray = None):
        if self.verbosity:
            print("Normalizing")
        # normalize data
        if self.normalize:
            m, n, k = data.shape
            data = StandardScaler().fit_transform(data.reshape(m * n, k)).reshape(m, n, k)
        self.data = data

        self.n_nearest = self._convert_to_int(self.n_nearest, data.shape[0])
        self.n_neighbors = self._convert_to_int(self.n_neighbors, data.shape[0])
        self.n_reference = self._convert_to_int(self.n_reference, data.shape[0])

        # compute distance matrix
        if ws_distance_matrix is None:
            if self.verbosity:
                print("Computing distance matrix")
            ws_distance_matrix = gen_dist_matrix(data, beta=self.beta)
        self.ws_distance_matrix = ws_distance_matrix

        UMAP = partial(umap.UMAP, n_neighbors=self.n_neighbors, metric="precomputed")

        self.nds = NeighborhoodDeviationScore(ws_distance_matrix, self.n_reference, self.n_nearest)
        self.embeddings = []
        self.scores = []
        self.errors = []

        if self.verbosity:
            print("Embedding data and computing scores")
        warnings.filterwarnings("ignore", message="using precomputed metric; inverse_transform will be unavailable")

        # compute 1D periodic embedding
        periodic_embedding = UMAP(n_components=1,
                                  output_metric=circle_metric,
                                  n_epochs=self.n_epochs_1d).fit_transform(ws_distance_matrix)
        periodic_embedding = np.vectorize(normalize_angle)(periodic_embedding)
        self.embeddings.append(periodic_embedding)
        score, error = self.nds.nds(periodic_embedding, output_metric=circle_metric_without_grad)
        self.scores.append(score)
        self.errors.append(error)

        # compute embeddings for higher dimensions
        for n_components in range(1, self.max_dim + 1):
            if n_components == 1:
                embedding = UMAP(n_components=n_components, n_epochs=self.n_epochs_1d).fit_transform(ws_distance_matrix)
            else:
                embedding = UMAP(n_components=n_components).fit_transform(ws_distance_matrix)
            self.embeddings.append(embedding)
            score, error = self.nds.nds(embedding)
            self.scores.append(score)
            self.errors.append(error)

        if self.verbosity:
            print("Computing the dimensionality")
        self.dimensionality = get_stop_point(self.scores, self.threshold)
