from typing import List

from utils import PhysExperiment

# Common experiments

Pendulum = PhysExperiment("pendulum", 1, plot_config=[(0, 1)], column_names=["angle", "angular velocity"])
HarmonicOscillator = PhysExperiment("harmonic_oscillator", 1, plot_config=[(0, 1)], column_names=["x", "x dot"])
DoublePendulum = PhysExperiment("double_pendulum", 1, plot_config=[(0, 1), (2, 3)],
                                column_names=["theta1", "theta2", "p1", "p2"])
CoupledOscillator = PhysExperiment("coupled_oscillator", 2, plot_config=[(0, 1), (2, 3)],
                                   column_names=["x1", "x2", "x1 dot", "x2 dot"])
KeplerProblem = PhysExperiment("kepler_problem", 3, plot_config=[(0, 1), (2, 3)], column_names=["x", "y", "px", "py"])

common_experiments: List[PhysExperiment] = [
    Pendulum,
    HarmonicOscillator,
    DoublePendulum,
    CoupledOscillator,
    KeplerProblem,
]
