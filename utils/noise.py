import numpy as np
import ai_prentice_wasserstein
from tqdm import tqdm


def add_noise(data, scale):
    return data + np.random.normal(scale=scale, size=(data.shape))


def prentices_with_noise(data, strength_arr=np.linspace(0, 2, 40)):
    scale = np.std(data.reshape(-1, data.shape[-1]), axis=0)
    prentices = []
    for strength in tqdm(strength_arr):
        ai_prentice = ai_prentice_wasserstein.DimensionalityPrentice(verbosity=0)
        ai_prentice.fit(add_noise(data, scale * strength))
        prentices.append(ai_prentice)
    return prentices


def plot_dimensionalities(ax, prentices, true_dim, strength_arr=np.linspace(0, 2, 40), max_strength=2):
    dim_arr = [prentice.dimensionality for prentice in prentices]
    ax.scatter(strength_arr, dim_arr, label="predicted dimensionality")
    ax.plot([0, max_strength], [true_dim, true_dim], label="true dimensionality", c="red")
    min_dim = min(np.min(dim_arr), true_dim)
    max_dim = max(np.max(dim_arr), true_dim)
    ax.set_yticks(np.arange(min_dim, max_dim + 1))
    ax.set_xlabel("strength of the noise")
    ax.set_ylabel("dimensionality")
    ax.legend()