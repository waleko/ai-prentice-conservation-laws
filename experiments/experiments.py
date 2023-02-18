
from typing import List

from utils import PhysExperiment

# Common experiments

Pendulum = PhysExperiment("pendulum", 1, plot_config=[(0, 1)])
HarmonicOscillator = PhysExperiment("harmonic_oscillator", 1, plot_config=[(0, 1)])
DoublePendulum = PhysExperiment("double_pendulum", 1, plot_config=[(0, 1), (2, 3)])
CoupledOscillator = PhysExperiment("coupled_oscillator", 2, plot_config=[(0, 1), (2, 3)])
KeplerProblem = PhysExperiment("kepler_problem", 3, plot_config=[(0, 1), (2, 3)])

common_experiments: List[PhysExperiment] = [
    Pendulum,
    HarmonicOscillator,
    DoublePendulum,
    CoupledOscillator,
    KeplerProblem,
]
