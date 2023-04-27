import utils
import umap
from functools import partial
from typing import Union
import numpy as np
from sklearn.preprocessing import StandardScaler


class DimensionalityPredictor:
    MAX_DIM = 6
    CHECK_PERIODIC = True

    def __init__(self, name_: str, normalize_: bool=True, beta_: float=2.0, n_neighbors_: Union[int, float]=80,
                 n_reference_: Union[int, float]=20, n_nearest_: Union[int, float]=40):
        self.name = name_
        self.normalize = normalize_
        self.beta = beta_
        self.n_neighbors = n_neighbors_
        self.n_reference = n_reference_
        self.n_nearest = n_nearest_

    def _convert_to_int(self, val, N):
        if type(val) == float:
            return int(val * N)
        return val

    def fit(self, data: np.ndarray, ws_distance_matrix: np.ndarray=None):
        
        print("Normalizing")
        if self.normalize:
            m, n, k = data.shape
            data = StandardScaler().fit_transform(data.reshape(m * n, k)).reshape(m, n, k)
        self.data = data

        self.n_nearest = self._convert_to_int(self.n_nearest, data.shape[0])
        self.n_neighbors = self._convert_to_int(self.n_neighbors, self.n_nearest)
        self.n_reference = self._convert_to_int(self.n_reference, data.shape[0])
        
        if ws_distance_matrix is None:
            print("Computing distance matrix")
            ws_distance_matrix = utils.gen_dist_matrix(data, beta=self.beta)
        self.ws_distance_matrix = ws_distance_matrix
        
        UMAP = partial(umap.UMAP, n_neighbors=self.n_neighbors, metric="precomputed")
        
        self.OrderScore = utils.OrderScore(ws_distance_matrix, self.n_reference, self.n_nearest)
        self.embeddings = []
        self.scores = []
        
        print("Embedding data and computing scores")
        if self.CHECK_PERIODIC:
            periodic_embedding = UMAP(n_components=1, output_metric=utils.circle_metric, n_epochs=10000).fit_transform(ws_distance_matrix)
            periodic_embedding = np.vectorize(utils.normalize_angle)(periodic_embedding)
            self.embeddings.append(periodic_embedding)
            self.scores.append(self.OrderScore.order_score(periodic_embedding, output_metric=utils.circle_metric_without_grad))
        for n_components in range(1, self.MAX_DIM + 1):
            if n_components == 1:
                embedding = UMAP(n_components=n_components, n_epochs=10000).fit_transform(ws_distance_matrix)
            else:
                embedding = UMAP(n_components=n_components).fit_transform(ws_distance_matrix)
            self.embeddings.append(embedding)
            self.scores.append(self.OrderScore.order_score(embedding))
        
        print("Computing the dimensionality")
        self.dimensionality = utils.get_stop_point(self.scores)

    def plot_scores(self, ax):
        ax.scatter(np.arange(1, self.MAX_DIM + 1), self.scores[1:], label="non-periodic embeddings")
        ax.scatter([1], [self.scores[0]], label="periodic embedding")
        ax.set_ylim([0, 1.1 * np.max(self.scores)])
        ax.set_xlabel("dimensionality of embedding")
        ax.set_ylabel("score")
        ax.legend()
        ax.set_title(self.name)

    def plot_score_diffs(self, ax):
        scores = np.concatenate(([np.min(self.scores[:2])], self.scores[2:]))
        score_diffs = scores[:-1] - scores[1:]
        
        x = list(range(1, self.dimensionality)) + list(range(self.dimensionality + 1, self.MAX_DIM))
        y = list(score_diffs[:self.dimensionality - 1]) + list(score_diffs[self.dimensionality:])
        ax.scatter(x, y)
        ax.scatter([self.dimensionality], [score_diffs[self.dimensionality - 1]], label="Point of the stopping")
        ax.set_xlabel("dimensionality of embedding")
        ax.set_ylabel("score difference")
        ax.legend()
        ax.set_title(self.name)

    def plot_1d(self, ax, conserved_quantity, quantity_name):
        if self.scores[2] + 0.01 < self.scores[1]:
            x = np.cos(self.embeddings[0])
            y = np.sin(self.embeddings[0])
            ax.scatter(x, y, c=conserved_quantity)
            ax.set_xlabel("$\\cos(embedding_i)$")
            ax.set_ylabel("$\\sin(embedding_i)$")
        else:
            ax.scatter(*self.embeddings[1].T, [0] * len(self.embeddings[1]), c=conserved_quantity)
            ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("embedding")
        ax.set_title(f"Colored by {quantity_name}")

    def plot_embedding_vs_conserved_quantity(self, ax, conserved_quantity, quantity_name):
        if self.scores[2] + 0.01 < self.scores[1]:
            ax.scatter(conserved_quantity, *self.embeddings[0].T)
        else:
            ax.scatter(conserved_quantity, *self.embeddings[1].T)
        ax.set_xlabel(quantity_name)
        ax.set_ylabel("embedding")

    def plot_2d(self, ax, conserved_quantity, quantity_name, embedding=None, i=1, j=2):
        if embedding is None:
            embedding = self.embeddings[2]
        ax.scatter(*embedding.T, c=conserved_quantity)
        ax.set_xlabel(f"component {i}")
        ax.set_ylabel(f"component {j}")
        ax.set_title(f"colored by {quantity_name}")

    def show_results(self, plt, *args):
        fig, axes = plt.subplots(1, 2, figsize=(17, 8))
        self.plot_scores(axes[0])
        self.plot_score_diffs(axes[1])
        plt.show()

        conserved_quantities, quantities_names = args
        if self.dimensionality == 1:
            fig, axes = plt.subplots(1, 2, figsize=(17, 8))
            self.plot_1d(axes[0], conserved_quantities[:, 0], quantities_names[0])
            self.plot_embedding_vs_conserved_quantity(axes[1], conserved_quantities[:, 0], quantities_names[0])

        if self.dimensionality == 2:
            fig, axes = plt.subplots(1, 2, figsize=(17, 8))
            for ax, conserved_quantity, quantity_name in zip(axes, conserved_quantities.T, quantities_names):
                self.plot_2d(ax, conserved_quantity, quantity_name)
        
        if self.dimensionality == 3:
            fig, axes = plt.subplots(3, 3, figsize=(17, 17))
            for axes_row, conserved_quantity, quantity_name in zip(axes, conserved_quantities.T, quantities_names):
                for ax, (i, j) in zip(axes_row, [(0, 1), (0, 2), (1, 2)]):
                    self.plot_2d(ax, conserved_quantity, quantity_name, self.embeddings[3][:, [i, j]], i + 1, j + 1)
    
