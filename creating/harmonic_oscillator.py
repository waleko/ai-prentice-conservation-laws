import numpy as np
import matplotlib.pyplot as plt
from . import auxiliary_functions as af


def random_points_on_unit_circle(size=200):
    random_angles = 2 * np.pi * np.random.uniform(size=size)
    return np.stack((np.sin(random_angles), np.cos(random_angles)), axis=1)


def single_trajectory(filename: str):
    r = np.random.uniform(0.1, 1)
    traj = r * random_points_on_unit_circle()
    np.savetxt(filename, traj)


def energy(state):
    x, p = state
    return p ** 2 / 2 + x ** 2 / 2


def create_trajectories(N_traj):
    af.create_multiple_trajectories("harmonic_oscillator", N_traj, single_trajectory)
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
