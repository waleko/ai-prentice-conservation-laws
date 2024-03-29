"""
Functions for plotting graphs
"""
from itertools import combinations
from typing import List, Tuple

import matplotlib.cm as cm
import numpy as np


def plot_scores(ax, scores, errors):
    """
    Plots NDS scores with error bars
    @param ax: matplotlib axis object
    @param scores: NDS scores
    @param errors: Errors of NDS scores
    """
    # get periodic embedding score
    score_periodic, scores = scores[:1], scores[1:]
    error_periodic, errors = errors[:1], errors[1:]
    # plot scores
    ax.errorbar(np.arange(len(scores)) + 1, scores, yerr=errors, label="non-periodic embeddings", c="black", fmt="o")
    ax.errorbar([0.9], score_periodic, yerr=error_periodic, label="periodic embedding", c="grey", fmt="o")
    # set axis limits
    max_error = np.max(error_periodic + errors)
    ax.set_ylim([1.1 * min(0, np.min(score_periodic + scores) - max_error),
                 1.1 * (np.max(score_periodic + scores) + max_error)])
    # set axis labels
    ax.set_xlabel("dimensionality of embedding")
    ax.set_ylabel("score")
    ax.legend(loc='upper right')


def plot_score_diffs(ax, scores, errors, dim_stop, threshold):
    """
    Plots the difference between NDS scores with error bars
    @param ax: matplotlib axis object
    @param scores: NDS scores
    @param errors: Errors of NDS scores
    @param dim_stop: Dimensionality of the embedding where the stopping criterion is met
    @param threshold: Threshold for stopping
    """
    if scores[0] < scores[1]:
        scores = np.array(scores[:1] + scores[2:])
        errors = np.array(errors[:1] + errors[2:])
    else:
        scores = np.array(scores[1:])
        errors = np.array(errors[1:])

    # calculate score differences
    score_diffs = scores[:-1] - scores[1:]
    score_diffs_errors = np.sqrt(errors[:-1] ** 2 + errors[1:] ** 2)

    # plot score differences
    x = list(range(1, dim_stop)) + list(range(dim_stop + 1, len(scores)))
    y = list(score_diffs[:dim_stop - 1]) + list(score_diffs[dim_stop:])
    y_errors = list(score_diffs_errors[:dim_stop - 1]) + list(score_diffs_errors[dim_stop:])
    ax.errorbar(x, y, yerr=y_errors, color="blue", fmt="o")
    ax.errorbar([dim_stop], [score_diffs[dim_stop - 1]], yerr=[score_diffs_errors[dim_stop - 1]],
                label="Point of the stopping", c="green", fmt="o")
    ax.plot([1, len(scores) - 1], [threshold, threshold], linestyle="--", label="threshold")
    ax.set_xlabel("dimensionality of embedding")
    ax.set_ylabel("score difference")
    ax.legend(loc='upper right')


def plot_1d(fig, ax, embedding, conserved_quantity, quantity_name):
    """
    Plots a 1D embedding
    @param fig: Figure object
    @param ax: Axis object
    @param embedding: 1D embedding
    @param conserved_quantity: Conserved quantity values
    @param quantity_name: Conserved quantity name
    """
    # plot embedding
    ax.scatter(*embedding.T, [0] * len(embedding), c=conserved_quantity)

    # remove axis
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("embedding")

    # colorbar
    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_array(conserved_quantity)

    fig.colorbar(sm, ax=ax)
    ax.set_title(f"Colored by {quantity_name}")


def plot_periodic_1d(fig, ax, embedding, conserved_quantity, quantity_name):
    """
    Plots a 1D periodic embedding
    @param fig: Figure object
    @param ax: Axis object
    @param embedding: 1D periodic embedding
    @param conserved_quantity: Conserved quantity values
    @param quantity_name: Conserved quantity name
    """
    # plot embedding
    x = np.cos(embedding.T)
    y = np.sin(embedding.T)
    ax.scatter(*x, *y, c=conserved_quantity)
    ax.set_xlabel("$\\cos$(embedding)")
    ax.set_ylabel("$\\sin$(embedding)")

    # colorbar
    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_array(conserved_quantity)

    fig.colorbar(sm, ax=ax)
    ax.set_title(f"Colored by {quantity_name}")


def plot_embedding_vs_conserved_quantity(ax, embedding, conserved_quantity, quantity_name):
    """
    Plots the embedding against the conserved quantity
    @param ax: Axis object
    @param embedding: 1D embedding
    @param conserved_quantity: Conserved quantity values
    @param quantity_name: Conserved quantity name
    """
    ax.scatter(conserved_quantity, *embedding.T, c="black")
    ax.set_xlabel(quantity_name)
    ax.set_ylabel("embedding")


def plot_2d(fig, ax, embedding, conserved_quantity, quantity_name, i=1, j=2):
    """
    Plots a 2D embedding
    @param fig: Figure object
    @param ax: Axis object
    @param embedding: 2D embedding
    @param conserved_quantity: Conserved quantity values
    @param quantity_name: Conserved quantity name
    @param i: Index of the first embedding component
    @param j: Index of the second embedding component
    """
    ax.scatter(*embedding.T, c=conserved_quantity)
    ax.set_xlabel(f"embedding component {i}")
    ax.set_ylabel(f"embedding component {j}")
    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_array(conserved_quantity)

    fig.colorbar(sm, ax=ax)
    ax.set_title(f"Colored by {quantity_name}")


def plot_all_2d(fig, axes, embedding, conserved_quantities, quantities_names):
    """
    Plots all 2D embeddings
    @param fig: Figure object
    @param axes: Axis objects
    @param embedding: 2D embedding
    @param conserved_quantities: Values of the conserved quantities
    @param quantities_names: Names of the conserved quantities
    """
    for ax, conserved_quantity, quantity_name in zip(axes, conserved_quantities.T, quantities_names):
        plot_2d(fig, ax, embedding, conserved_quantity, quantity_name)


def plot_all_3d(fig, axes, embedding, conserved_quantities, quantities_names):
    """
    Plots all 3D embeddings
    @param fig: Figure object
    @param axes: Axis objects
    @param embedding: 3D embedding
    @param conserved_quantities: Values of the conserved quantities
    @param quantities_names: Names of the conserved quantities
    """
    for axes_row, conserved_quantity, quantity_name in zip(axes, conserved_quantities.T, quantities_names):
        for ax, (i, j) in zip(axes_row, [(0, 1), (1, 2), (0, 2)]):
            plot_2d(fig, ax, embedding[:, [i, j]], conserved_quantity, quantity_name, i + 1, j + 1)


def choose_coordinates(embedding, quantities, n_coords=2) -> List[Tuple[int, ...]]:
    """
    Chooses the best coordinates for each conserved quantity
    @param embedding: Embedding
    @param quantities: Conserved quantities values
    @param n_coords: Number of coordinates to choose
    @return: Best coordinates for each conserved quantity
    """
    best_coordinates = []
    for quantity in quantities.T:
        best_coords = None
        best_mse = np.inf
        for coords in combinations(range(embedding.shape[1]), n_coords):
            X = embedding[:, list(coords)]
            mse = ((np.linalg.multi_dot((X, np.linalg.inv((X.T.dot(X))), X.T, quantity)) - quantity) ** 2).mean()
            if mse < best_mse:
                best_mse = mse
                best_coords = coords
        best_coordinates.append(best_coords)
    return best_coordinates
