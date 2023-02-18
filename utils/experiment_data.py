import logging
from typing import Optional

from .data_loader import *


class PhysExperiment:
    def __init__(self, experiment_name: str, n_conservation_laws: int, traj_cnt: int = 200, traj_len: int = 1000,
                 plot_config: Optional[list[tuple[int, int]]] = None, start_index: int = 0):
        self.experiment_name = experiment_name
        self.n_conservation_laws = n_conservation_laws
        self.traj_cnt = traj_cnt
        self.traj_len = traj_len
        self.data = get_data(experiment_name, traj_cnt, traj_len, start_index=start_index)
        self.plot_config = plot_config

    def single_trajectory(self, random_seed=None):
        np.random.seed(random_seed)
        idx = np.random.randint(0, self.traj_cnt)
        return self.data[idx]

    def plot_trajectory(self, traj: np.ndarray):
        if self.plot_config is None:
            logging.warning(f"plot config for {self.experiment_name} is not specified")
            return

        # plot for every configuration
        N = len(self.plot_config)
        fig, axs = plt.subplots(N)
        fig.suptitle(f"Input trajectories: {self.experiment_name}")
        for idx in range(N):
            # components on the plot
            j, k = self.plot_config[idx]
            if N == 1:
                x = axs
            else:
                x = axs[idx]
            x.scatter(traj[:, j], traj[:, k])
            x.set_xlabel(f"Component #{j}")
            x.set_ylabel(f"Component #{k}")
        fig.show()

    @property
    def pt_dim(self):
        return self.data.shape[-1]

    @property
    def n_eff(self):
        return self.pt_dim - self.n_conservation_laws
