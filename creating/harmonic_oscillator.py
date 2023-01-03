import numpy as np
import matplotlib.pyplot as plt
from . import auxiliary_functions as af


def single_trajectory(filename: str, params: tuple):
    t = 2 * np.pi * np.random.uniform(size=200)
    np.savetxt(filename, np.stack((np.sin(t), np.cos(t))).transpose() * params[0])


def energy(state):
    return (state ** 2).sum() / 2


def create_trajectories(N_traj):
    af.create_multiple_trajectories("harmonic_oscillator", N_traj, single_trajectory, zip(np.random.uniform(0.1, 1, size=N_traj)))
    af.add_noise("harmonic_oscillator", N_traj)
    plt.figure()
    for i in range(10):
        traj = af.read_traj("harmonic_oscillator", i)
        plt.scatter(traj[:, 0], traj[:, 1], s=1)
    plt.xlabel("x")
    plt.ylabel("p")
    plt.title("some harmonic oscillator trajectories")
    plt.savefig("trajectories/harmonic_oscillator/some_trajectories.png")
    af.compute_conserved_quantity("harmonic_oscillator", "E", energy, N_traj)
    af.normalize("harmonic_oscillator", N_traj)
