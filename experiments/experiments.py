import numpy as np

from typing import List

import utils
from utils import PhysExperiment, CsvPhysExperiment
from creating import double_pendulum as dp
from .animations import *


def rand_point(dim=5):
    x = np.random.rand(dim) * 2 - 1
    return x / np.linalg.norm(x)


def rel_deviation(traj_dp):
    energies1_all = np.array([dp.energy1(x) for x in traj_dp])
    rel_std = np.std(energies1_all) / np.mean(energies1_all)
    return rel_std


# Common experiments
Pendulum = CsvPhysExperiment("pendulum", 1, plot_config=[(0, 1)], column_names=["angle", "angular velocity"],
                             trajectory_animator=pendulum_animator)
HarmonicOscillator = CsvPhysExperiment("harmonic_oscillator", 1, plot_config=[(0, 1)], column_names=["x", "x dot"],
                                       trajectory_animator=harmonic_oscillator_animator)
# DoublePendulum = CsvPhysExperiment("double_pendulum", 1, plot_config=[(0, 1), (2, 3)],
#                                    column_names=["theta1", "theta2", "p1", "p2"])
DoublePendulumLowEnergy = CsvPhysExperiment("double_pendulum_low_energy", 2, plot_config=[(0, 1), (2, 3)],
                                            column_names=["theta1", "theta2", "p1", "p2"],
                                            trajectory_animator=double_pendulum_animator)
DoublePendulumHighEnergy = CsvPhysExperiment("double_pendulum_high_energy", 1, plot_config=[(0, 1), (2, 3)],
                                             column_names=["theta1", "theta2", "p1", "p2"],
                                             trajectory_animator=double_pendulum_animator)
CoupledOscillator = CsvPhysExperiment("coupled_oscillator", 2, plot_config=[(0, 1), (2, 3)],
                                      column_names=["x1", "x2", "x1 dot", "x2 dot"],
                                      trajectory_animator=coupled_oscillator_animator)
KeplerProblem = CsvPhysExperiment("kepler_problem", 3, plot_config=[(0, 1), (2, 3)],
                                  column_names=["x", "y", "px", "py"], trajectory_animator=kepler_problem_animator)
# Utility experiments
Sphere5 = PhysExperiment("sphere5", 1, np.array(
    [np.array([rand_point(dim=5) for _ in range(1000)]) * np.random.randint(1, 100) for _ in range(200)]))
# Advanced
# Kdv = PhysExperiment("KdV (infinitely many conserved quantities)", 3, np.load("trajectories/kdv_400_200_200_ver2.npz"), column_names=[])

common_experiments: List[PhysExperiment] = [
    Pendulum,
    HarmonicOscillator,
    # DoublePendulum,
    DoublePendulumLowEnergy,
    DoublePendulumHighEnergy,
    CoupledOscillator,
    KeplerProblem,
    Sphere5
]
