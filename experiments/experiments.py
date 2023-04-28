from typing import List

from sklearn.preprocessing import MaxAbsScaler

from utils import PhysExperiment, NpzPhysExperiment
from .animations import *


def rand_point(dim=5):
    x = np.random.rand(dim) * 2 - 1
    return x / np.linalg.norm(x)


def rand_cylinder_point():
    angle = np.random.rand() * np.pi
    z = np.random.rand()
    return np.array([np.cos(angle), np.sin(angle), z])


def preprocess_turing(data):
    data = data.reshape(*data.shape[:-1], 2, data.shape[-1] // 2).swapaxes(-2, -1).reshape(data.shape)
    scaled_data = MaxAbsScaler().fit_transform(data.reshape(-1, 2)).reshape(data.shape)
    return scaled_data


def preprocess_kdv(data):
    data = np.stack((data, np.roll(data, -1, axis=2) - data), axis=3)
    scaled_data = MaxAbsScaler().fit_transform(data.reshape(-1, 2)).reshape(data.shape)
    return scaled_data


# Common experiments
Pendulum = NpzPhysExperiment("pendulum", 1, plot_config=[(0, 1)], column_names=["angle", "angular velocity"],
                             trajectory_animator=pendulum_animator)
HarmonicOscillator = NpzPhysExperiment("harmonic_oscillator", 1, plot_config=[(0, 1)], column_names=["x", "x dot"],
                                       trajectory_animator=harmonic_oscillator_animator)
DoublePendulumLowEnergy = NpzPhysExperiment("double_pendulum_low_energy", 2, plot_config=[(0, 2), (1, 3)],
                                            column_names=["theta1", "theta2", "p1", "p2"],
                                            trajectory_animator=double_pendulum_animator)
DoublePendulumHighEnergy = NpzPhysExperiment("double_pendulum_high_energy", 1, plot_config=[(0, 2), (1, 3)],
                                             column_names=["theta1", "theta2", "p1", "p2"],
                                             trajectory_animator=double_pendulum_animator)
CoupledOscillator = NpzPhysExperiment("coupled_oscillator", 2, plot_config=[(0, 2), (1, 3)],
                                      column_names=["x1", "x2", "x1 dot", "x2 dot"],
                                      trajectory_animator=coupled_oscillator_animator)
KeplerProblem = NpzPhysExperiment("kepler_problem", 3, plot_config=[(0, 1), (2, 3)],
                                  column_names=["x", "y", "px", "py"], trajectory_animator=kepler_problem_animator)
# Utility experiments
Sphere3 = PhysExperiment("sphere3", 1, np.array(
    [np.array([rand_point(dim=3) for _ in range(1000)]) * np.random.randint(1, 100) for _ in range(200)]))
Sphere5 = PhysExperiment("sphere5", 1, np.array(
    [np.array([rand_point(dim=5) for _ in range(1000)]) * np.random.randint(1, 100) for _ in range(200)]))
Cylinder = PhysExperiment("cylinder", 1, np.array(
    [np.array([rand_cylinder_point() for _ in range(1000)]) * np.random.randint(1, 100) for _ in range(200)]))

# Advanced
Turing = NpzPhysExperiment("turing", 1, preprocess=preprocess_turing, filename="turing_400_200_50_l8.0")
KdV = NpzPhysExperiment("kdv", 3, preprocess=preprocess_kdv, filename="kdv_400_200_200_ver2")

common_experiments: List[PhysExperiment] = [
    Pendulum,
    HarmonicOscillator,
    DoublePendulumLowEnergy,
    DoublePendulumHighEnergy,
    CoupledOscillator,
    KeplerProblem
]
