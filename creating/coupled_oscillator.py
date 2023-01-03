import numpy as np
import matplotlib.pyplot as plt

import auxiliary_functions as af


def energy(state):
    x1, x2, p1, p2 = state
    return x1 ** 2 + x2 ** 2 - x1 * x2 + (p1 ** 2 + p2 ** 2) / 2


def energy1(state):
    x1, x2, p1, p2 = state
    return ((x1 + x2) ** 2 + (p1 + p2) ** 2) / 4


def energy2(state):
    x1, x2, p1, p2 = state
    return (3 * (x1 - x2) ** 2 + (p1 - p2) ** 2) / 4


def single_trajectory(filename, params):
    E1, E2 = params
    A = np.sqrt(E1)
    B = np.sqrt(E2) / 2
    t = np.random.uniform(0, 100, size=1000)
    mode1 = A * np.stack([np.cos(t), np.cos(t), -np.sin(t), -np.sin(t)]).transpose()
    mode2 = B * np.stack([np.cos(np.sqrt(3) * t), - np.cos(np.sqrt(3) * t), -np.sin(np.sqrt(3) * t), np.sin(np.sqrt(3) * t)]).transpose()
    np.savetxt(filename, mode1 + mode2)


def create_trajectories(N_traj):
    af.create_multiple_trajectories("coupled_oscillator", N_traj, single_trajectory, zip(np.random.uniform(0, 1, size=N_traj), np.random.uniform(0, 1, size=N_traj)))
    af.add_noise("coupled_oscillator", N_traj)
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(45, 15)
    for i in range(3):
        traj = af.read_traj("coupled_oscillator", i)
        x1, x2, p1, p2 = traj.transpose()
        axes[i].scatter(x1, p1, s=5)
        axes[i].scatter(x2, p2, s=5)
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("p")
    axes[1].set_title("some coupled oscillator trajectories\ncolored by objects")
    plt.savefig("../trajectories/coupled_oscillator/some_trajectories.png")
    af.compute_conserved_quantity("coupled_oscillator", "E", energy)
    af.compute_conserved_quantity("coupled_oscillator", "E1", energy1)
    af.compute_conserved_quantity("coupled_oscillator", "E2", energy2)
    af.normalize("coupled_oscillator", N_traj)
