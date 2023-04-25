import logging
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt, animation
from torch import nn

from utils import PhysExperiment


class Animator:
    def __init__(self, exp: PhysExperiment, exp_full_name: Optional[str] = None):
        if exp_full_name is None:
            exp_full_name = exp.experiment_name
        self.__tensor_traj = None
        self.exp = exp
        self.exp_full_name = exp_full_name
        self.frames = []
        self.orig_traj = None
        self.run_name = None

    def start(self, traj: torch.Tensor, run_name: Optional[str] = None):
        if run_name is None:
            run_name = self.exp_full_name
        else:
            run_name = f"{self.exp_full_name}_{run_name}"
        self.frames = []
        self.__tensor_traj = traj
        self.run_name = run_name
        self.orig_traj = traj.detach().cpu().numpy()

    def log(self, transformed_traj: np.ndarray):
        self.frames.append(transformed_traj)

    def save(self):
        if self.exp.plot_config is None:
            logging.warning(f"plot config for {self.exp.experiment_name} is not specified")
            return

        # fig = plt.figure()
        # plot for every configuration
        N = len(self.exp.plot_config)
        fig, axs = plt.subplots(N, figsize=(8, 4 * (N + 1)))
        fig.suptitle(f"Experiment: {self.run_name}")
        scatters = []

        for idx in range(N):
            # components on the plot
            j, k = self.exp.plot_config[idx]
            if N == 1:
                x = axs
            else:
                x = axs[idx]
            scatters.append(x.scatter(self.orig_traj[:, j], self.orig_traj[:, k], c='blue', s=1))
            scatters.append(x.scatter([], [], c='orange', s=1))
            x.set_xlabel(self.exp.column_names[j])
            x.set_ylabel(self.exp.column_names[k])

        def update_lines(num):
            for idx in range(N):
                j, k = self.exp.plot_config[idx]
                scatters[idx * 2 + 1].set_offsets(self.frames[num - 1][:, (j, k)]) # only orange
            return scatters

        # Creating the Animation object
        ani = animation.FuncAnimation(fig, update_lines, len(self.frames), interval=100)
        fname = f"animation_{self.run_name}.mp4"
        ani.save(fname)

        return fname

    def forward_and_log(self, model: nn.Module):
        res = model(self.__tensor_traj).detach().cpu().numpy()
        self.log(res)
