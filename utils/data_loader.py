import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import pandas as pd

def get_data(experiment_name: str, count: int = 200, sample_size: int = 200, sep: str = ' ', n_plot: int = 5, plot_config: Union[List[Tuple[int, int]], None] = None):
    ans = []
    for i in range(count):
        d = pd.read_csv(f"trajectories/{experiment_name}/{i}.csv", sep=sep, header=None)
        N = len(d)
        assert N >= sample_size
        ans.append(d.to_numpy()[::(N // sample_size)])
    X = np.array(ans)
    # Plot
    if plot_config is not None:
        X_plot = X[:n_plot]
        N = len(plot_config)
        fig, axs = plt.subplots(N)
        fig.suptitle(f"Input trajectories: {experiment_name}")
        for idx in range(N):
            j, k = plot_config[idx]
            if N == 1:
                x = axs
            else:
                x = axs[idx]
            for traj in X_plot:
                x.scatter(traj[:,j], traj[:,k])
            x.set_xlabel(f"Component #{j}")
            x.set_ylabel(f"Component #{k}")
        fig.show()
    return X

def sortby_H(X: np.ndarray, H) -> np.ndarray:
    """
    Sorts data by Hamiltonian
    
    Makes it possible to plot beautiful distance matrices
    """
    Hs = np.array([H(x[0]) for x in X])
    return X[Hs.argsort()]