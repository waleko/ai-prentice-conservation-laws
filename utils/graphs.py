import numpy as np
import matplotlib.cm as cm
from itertools import combinations


def plot_scores(ax, scores, errors):
    score_periodic, scores = scores[:1], scores[1:]
    error_periodic, errors = errors[:1], errors[1:]
    ax.errorbar(np.arange(len(scores)) + 1, scores, yerr=errors, label="non-periodic embeddings", c="black", fmt="o")
    ax.errorbar([0.9], score_periodic, yerr=error_periodic, label="periodic embedding", c="grey", fmt="o")
    max_error = np.max(error_periodic + errors)
    ax.set_ylim([1.1 * min(0, np.min(score_periodic + scores) - max_error), 1.1 * (np.max(score_periodic + scores) + max_error)])
    ax.set_xlabel("dimensionality of embedding")
    ax.set_ylabel("score")
    ax.legend(loc='upper right')


def plot_score_diffs(ax, scores, errors, dim_stop, threshold):
    if scores[0] < scores[1]:
        scores = np.array(scores[:1] + scores[2:])
        errors = np.array(errors[:1] + errors[2:])
    else:
        scores = np.array(scores[1:])
        errors = np.array(errors[1:])

    score_diffs = scores[:-1] - scores[1:]
    score_diffs_errors = np.sqrt(errors[:-1] ** 2 + errors[1:] ** 2)
    
    x = list(range(1, dim_stop)) + list(range(dim_stop + 1, len(scores)))
    y = list(score_diffs[:dim_stop - 1]) + list(score_diffs[dim_stop:])
    y_errors = list(score_diffs_errors[:dim_stop - 1]) + list(score_diffs_errors[dim_stop:])
    ax.errorbar(x, y, yerr=y_errors, color="blue", fmt="o")
    ax.errorbar([dim_stop], [score_diffs[dim_stop - 1]], yerr=[score_diffs_errors[dim_stop - 1]], label="Point of the stopping", c="green", fmt="o")
    ax.plot([1, len(scores) - 1], [threshold, threshold], linestyle="--", label="threshold")
    ax.set_xlabel("dimensionality of embedding")
    ax.set_ylabel("score difference")
    ax.legend(loc='upper right')


def plot_1d(fig, ax, embedding, conserved_quantity, quantity_name):
    ax.scatter(*embedding.T, [0] * len(embedding), c=conserved_quantity)

    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("embedding")
    
    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_array(conserved_quantity)

    fig.colorbar(sm, ax=ax)
    ax.set_title(f"Colored by {quantity_name}")


def plot_periodic_1d(fig, ax, embedding, conserved_quantity, quantity_name):
    x = np.cos(embedding.T)
    y = np.sin(embedding.T)
    ax.scatter(*x, *y, c=conserved_quantity)
    ax.set_xlabel("$\\cos$(embedding)")
    ax.set_ylabel("$\\sin$(embedding)")

    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_array(conserved_quantity)

    fig.colorbar(sm, ax=ax)
    ax.set_title(f"Colored by {quantity_name}")


def plot_embedding_vs_conserved_quantity(ax, embedding, conserved_quantity, quantity_name):
    ax.scatter(conserved_quantity, *embedding.T, c="black")
    ax.set_xlabel(quantity_name)
    ax.set_ylabel("embedding")


def plot_2d(fig, ax, data, c, c_name, i=1, j=2, plotted="embedding"):
    ax.scatter(*data.T, c=c)
    ax.set_xlabel(f"{plotted} component {i}")
    ax.set_ylabel(f"{plotted} component {j}")
    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_array(c)

    fig.colorbar(sm, ax=ax)
    ax.set_title(f"Colored by {c_name}")


def plot_all_2d(fig, axes, embedding, conserved_quantities, quantities_names):
    for ax, conserved_quantity, quantity_name in zip(axes, conserved_quantities.T, quantities_names):
        plot_2d(fig, ax, embedding, conserved_quantity, quantity_name)


def plot_all_3d(fig, axes, embedding, conserved_quantities, quantities_names):
    for axes_row, conserved_quantity, quantity_name in zip(axes, conserved_quantities.T, quantities_names):
        for ax, (i, j) in zip(axes_row, [(0, 1), (1, 2), (0, 2)]):
            plot_2d(fig, ax, embedding[:, [i, j]], conserved_quantity, quantity_name, i + 1, j + 1)
            
            
def choose_coordinates(embedding, quantities, n_coords=2):
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


def plot_dynamics(plt, data, true_dynamics, predicted_dynamics):
    if true_dynamics.shape[1] == 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(true_dynamics, predicted_dynamics)
        axes[0].set_xlabel("true dynamics")
        axes[0].set_ylabel("predicted dynamics")
        plot_2d(fig, axes[1], data[:, :2], predicted_dynamics.T[0], "predicted dynamics", plotted="data")
        
    elif true_dynamics.shape[1] == 2:
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        for i in range(2):
            for j in range(2):
                axes[j, i].scatter(true_dynamics[:, j], predicted_dynamics[:, i])
                axes[j, i].set_xlabel(f"true dynamics component #{j + 1}")
                axes[j, i].set_ylabel(f"predicted dynamics component #{i + 1}")
            plot_2d(fig, axes[2, i], true_dynamics, predicted_dynamics[:, i], f"predicted dynamics component #{i + 1}", plotted="true dynamics")

    elif true_dynamics.shape[1] == 3:
        pass
    
    else:
        print("Not supported number of dynamical components")
        
    return fig, axes