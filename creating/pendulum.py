import numpy as np
import matplotlib.pyplot as plt
from . import auxiliary_functions as af


def single_trajectory(filename: str):
    state0_generator = lambda: np.array([0, np.sqrt(np.random.uniform(0.1, 3))])

    def derivative(arr):
        return np.array([arr[1], -np.sin(arr[0])])
    
    af.run_and_write(derivative, state0_generator, filename, energy, "absolute", 0.1, 10)


def energy(state):
    theta, L = state
    return L ** 2 / 2 - np.cos(theta)


def create_trajectories(N_traj=200):
    af.create_multiple_trajectories("pendulum", N_traj, single_trajectory)
    af.add_noise("pendulum", N_traj)
    plt.figure()
    for i in range(10):
        traj = af.read_traj("pendulum", i)
        plt.scatter(np.vectorize(lambda x: x if x < np.pi else x - 2 * np.pi)(traj[:, 0] % (2 * np.pi)), traj[:, 1], s=1)
    plt.xlabel("theta")
    plt.ylabel("L")
    plt.title("some pendulum trajectories")
    plt.savefig("trajectories/pendulum/some_trajectories.png")
    af.compute_conserved_quantity("pendulum", "E", energy, N_traj)
    af.normalize("pendulum", N_traj)
