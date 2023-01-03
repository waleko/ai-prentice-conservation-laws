import numpy as np
from typing import List, Tuple, Union, Callable
import matplotlib.pyplot as plt
import pandas as pd


def get_data(experiment_name: str, count: int = 200, sample_size: int = 200, sep: str = ' ',
             n_plotted_trajectories: int = 5,
             plot_config: Union[List[Tuple[int, int]], None] = None) -> np.ndarray:
    """
    Gets trajectories data for the experiment
    @param experiment_name:
    @param count: Number of trajectories to read
    @param sample_size: Number of points in each trajectory
    @param sep: CSV separator
    @param n_plotted_trajectories:
    @param plot_config: Plots configurations: pairs of component indexes to plot
    @return: Loaded trajectories as a numpy array
    """
    ans = []
    for i in range(count):
        d = pd.read_csv(f"trajectories/{experiment_name}/{i}.csv", sep=sep, header=None)
        N = len(d)
        assert N >= sample_size
        # take evenly spaced `sample_size` points
        ans.append(d.to_numpy()[::(N // sample_size)])
    X = np.array(ans)

    # Optionally, plot input trajectories
    if plot_config is not None:
        # Plot only few trajectories
        X_plot = X[:n_plotted_trajectories]
        # plot for every configuration
        N = len(plot_config)
        fig, axs = plt.subplots(N)
        fig.suptitle(f"Input trajectories: {experiment_name}")
        for idx in range(N):
            # components on the plot
            j, k = plot_config[idx]
            if N == 1:
                x = axs
            else:
                x = axs[idx]
            for traj in X_plot:
                x.scatter(traj[:, j], traj[:, k])
            x.set_xlabel(f"Component #{j}")
            x.set_ylabel(f"Component #{k}")
        fig.show()
    return X

def sortby_H(X: np.ndarray, H: Callable[[np.ndarray], float]) -> np.ndarray:
    """
    Sorts data by Hamiltonian. Makes it possible to plot beautiful distance matrices.
    @param X: Trajectories data
    @param H: Hamiltonian
    @return: Sorted trajectories data
    """
    Hs = np.array([H(x[0]) for x in X])
    return X[Hs.argsort()]
