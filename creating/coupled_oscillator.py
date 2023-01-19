import numpy as np
from numpy import cos, sin, sqrt
import matplotlib.pyplot as plt

from . import auxiliary_functions as af


def energy(state):
    x1, x2, p1, p2 = state
    return x1 ** 2 + x2 ** 2 - x1 * x2 + (p1 ** 2 + p2 ** 2) / 2


def energy1(state):
    x1, x2, p1, p2 = state
    return ((x1 + x2) ** 2 + (p1 + p2) ** 2) / 4


def energy2(state):
    x1, x2, p1, p2 = state
    return (3 * (x1 - x2) ** 2 + (p1 - p2) ** 2) / 4


def single_trajectory(filename):
    E1, E2 = np.random.uniform(0, 1, size=2)
    A = np.sqrt(E1)
    B = np.sqrt(E2) / 2
    t = np.random.uniform(0, 100, size=1000)
    
    x1_arr_mode1 = x2_arr_mode1 = cos(t) * A
    p1_arr_mode1 = p2_arr_mode1 = -sin(t) * A

    x1_arr_mode2 = B * cos(sqrt(3) * t)
    x2_arr_mode2 = -B * cos(sqrt(3) * t)
    p1_arr_mode2 = -B * sin(sqrt(3) * t)
    p2_arr_mode2 = B * sin(sqrt(3) * t)

    x1_arr = x1_arr_mode1 + x1_arr_mode2
    x2_arr = x2_arr_mode1 + x2_arr_mode2
    p1_arr = p1_arr_mode1 + p1_arr_mode2
    p2_arr = p2_arr_mode1 + p2_arr_mode2

    traj = np.stack((x1_arr, x2_arr, p1_arr, p2_arr), axis=1)

    np.savetxt(filename, traj)


def create_trajectories(N_traj):
    af.create_multiple_trajectories("coupled_oscillator", N_traj, single_trajectory)
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
    plt.savefig("trajectories/coupled_oscillator/some_trajectories.png")
    af.compute_conserved_quantity("coupled_oscillator", "E", energy)
    af.compute_conserved_quantity("coupled_oscillator", "E1", energy1)
    af.compute_conserved_quantity("coupled_oscillator", "E2", energy2)
    af.normalize("coupled_oscillator", N_traj)
