import numpy as np
import ai_prentice_wasserstein
from tqdm import tqdm


def prentices_with_noise(data, strength_arr=np.linspace(0, 2, 40)):
    """
    Returns a list of prentices for different strengths of the noise
    @param data: Experiment data
    @param strength_arr: Noise strengths
    """
    scale = np.std(data.reshape(-1, data.shape[-1]), axis=0)
    prentices = []
    for strength in tqdm(strength_arr):
        ai_prentice = ai_prentice_wasserstein.DimensionalityPrentice(verbosity=0)
        ai_prentice.fit(data + np.random.normal(scale=scale * strength, size=(data.shape)))
        prentices.append(ai_prentice)
    return prentices


def plot_dimensionalities(ax, prentices, true_dim, strength_arr=np.linspace(0, 2, 40), max_strength=2):
    """
    Plots predicted dimensionalities for different strengths of the noise
    @param ax: Axis object
    @param prentices: Prentices for different strengths of the noise
    @param true_dim: True dimensionality of the data
    @param strength_arr: Strengths of the noise
    @param max_strength: Maximum strength of the noise
    """
    vals = [prentice.dimensionality for prentice in prentices]
    ax.yaxis.set_ticks(range(min(vals), max(vals) + 1))
    ax.scatter(strength_arr, vals, label="predicted dimensionality")
    ax.plot([0, max_strength], [true_dim, true_dim], label="true dimensionality", c="red")
    ax.set_xlabel("strength of the noise")
    ax.set_ylabel("dimensionality")
    ax.legend()
