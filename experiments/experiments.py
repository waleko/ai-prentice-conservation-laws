import numpy as np

from typing import List

import utils
from utils import PhysExperiment, CsvPhysExperiment
from creating import double_pendulum as dp


def rand_point(dim=5):
    x = np.random.rand(dim) * 2 - 1
    return x / np.linalg.norm(x)


def rel_deviation(traj_dp):
    energies1_all = np.array([dp.energy1(x) for x in traj_dp])
    rel_std = np.std(energies1_all) / np.mean(energies1_all)
    return rel_std


# Common experiments
Pendulum = CsvPhysExperiment("pendulum", 1, plot_config=[(0, 1)], column_names=["angle", "angular velocity"])
HarmonicOscillator = CsvPhysExperiment("harmonic_oscillator", 1, plot_config=[(0, 1)], column_names=["x", "x dot"])
# DoublePendulum = CsvPhysExperiment("double_pendulum", 1, plot_config=[(0, 1), (2, 3)],
#                                    column_names=["theta1", "theta2", "p1", "p2"])
__dp_data = utils.get_data("double_pendulum", 200, 1000)
__rel_dev = []
for traj in __dp_data:
    __rel_dev.append(rel_deviation(traj))
small_energy_indices = np.array(__rel_dev) < 0.06  # threshold

DoublePendulumSmallEnergy = PhysExperiment("double_pendulum_small_energy", 2, __dp_data[small_energy_indices],
                                           ["theta1", "theta2", "p1", "p2"], plot_config=[(0, 1), (2, 3)])
DoublePendulumLargeEnergy = PhysExperiment("double_pendulum_large_energy", 1, __dp_data[~small_energy_indices],
                                           ["theta1", "theta2", "p1", "p2"], plot_config=[(0, 1), (2, 3)])
CoupledOscillator = CsvPhysExperiment("coupled_oscillator", 2, plot_config=[(0, 1), (2, 3)],
                                      column_names=["x1", "x2", "x1 dot", "x2 dot"])
KeplerProblem = CsvPhysExperiment("kepler_problem", 3, plot_config=[(0, 1), (2, 3)],
                                  column_names=["x", "y", "px", "py"])
# Utility experiments
Sphere5 = PhysExperiment("sphere5", 1, np.array(
    [np.array([rand_point(dim=5) for _ in range(1000)]) * np.random.randint(1, 100) for _ in range(200)]))
# Advanced
# Kdv = PhysExperiment("KdV (infinitely many conserved quantities)", 3, np.load("trajectories/kdv_400_200_200_ver2.npz"), column_names=[])

common_experiments: List[PhysExperiment] = [
    Pendulum,
    HarmonicOscillator,
    # DoublePendulum,
    DoublePendulumSmallEnergy,
    DoublePendulumLargeEnergy,
    CoupledOscillator,
    KeplerProblem,
    Sphere5
]
