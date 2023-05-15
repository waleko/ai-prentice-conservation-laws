import numpy as np
import matplotlib.cm as cm


def plot_scores(ax, scores, errors):
    score_periodic, scores = scores[:1], scores[1:]
    error_periodic, errors = errors[:1], errors[1:]
    ax.errorbar(np.arange(len(scores)) + 1, scores, yerr=errors, label="non-periodic embeddings", c="black", fmt="o")
    ax.errorbar([0.9], score_periodic, yerr=error_periodic, label="periodic embedding", c="grey", fmt="o")
    max_error = np.max(error_periodic + errors)
    ax.set_ylim([1.1 * min(0, np.min(score_periodic + scores) - max_error), 1.1 * (np.max(score_periodic + scores) + max_error)])
    ax.set_xlabel("dimensionality of embedding", fontsize=15)
    ax.set_ylabel("score", fontsize=15)
    ax.legend(fontsize=15)


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
    ax.set_xlabel("dimensionality of embedding", fontsize=15)
    ax.set_ylabel("score difference", fontsize=15)
    ax.legend(fontsize=15)


def plot_1d(fig, ax, embedding, conserved_quantity, quantity_name):
    ax.scatter(*embedding.T, [0] * len(embedding), c=conserved_quantity)

    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("embedding", fontsize=15)
    
    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_array(conserved_quantity)

    fig.colorbar(sm, ax=ax)
    ax.set_title(f"Colored by {quantity_name}", fontsize=20)


def plot_periodic_1d(fig, ax, embedding, conserved_quantity, quantity_name):
    x = np.cos(embedding.T)
    y = np.sin(embedding.T)
    ax.scatter(*x, *y, c=conserved_quantity)
    ax.set_xlabel("$\\cos$(embedding)", fontsize=15)
    ax.set_ylabel("$\\sin$(embedding)", fontsize=15)

    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_array(conserved_quantity)

    fig.colorbar(sm, ax=ax)
    ax.set_title(f"Colored by {quantity_name}", fontsize=20)


def plot_embedding_vs_conserved_quantity(ax, embedding, conserved_quantity, quantity_name):
    ax.scatter(conserved_quantity, *embedding.T, c="black")
    ax.set_xlabel(quantity_name, fontsize=15)
    ax.set_ylabel("embedding", fontsize=15)


def plot_2d(fig, ax, embedding, conserved_quantity, quantity_name, i=1, j=2):
    ax.scatter(*embedding.T, c=conserved_quantity)
    ax.set_xlabel(f"embedding component {i}", fontsize=15)
    ax.set_ylabel(f"embedding component {j}", fontsize=15)
    sm = cm.ScalarMappable(cmap='viridis')
    sm.set_array(conserved_quantity)

    fig.colorbar(sm, ax=ax)
    ax.set_title(f"Colored by {quantity_name}", fontsize=20)


def plot_all_2d(fig, axes, embedding, conserved_quantities, quantities_names):
    for ax, conserved_quantity, quantity_name in zip(axes, conserved_quantities.T, quantities_names):
        plot_2d(fig, ax, embedding, conserved_quantity, quantity_name)


def plot_all_3d(fig, axes, embedding, conserved_quantities, quantities_names):
    for axes_row, conserved_quantity, quantity_name in zip(axes, conserved_quantities.T, quantities_names):
        for ax, (i, j) in zip(axes_row, [(0, 1), (1, 2), (0, 2)]):
            plot_2d(fig, ax, embedding[:, [i, j]], conserved_quantity, quantity_name, i + 1, j + 1)