import numpy as np
from .auxiliary_functions import *


def energy(state):
    theta, L = state
    return L ** 2 / 2 - np.cos(theta) + 1


def create_trajectories(N_traj, traj_len=200, normalize=True, save=True):
    """
    Creates trajectories of pendulum with different energies. Returns trajectories and energies
    @param N_traj: number of created trajectories
    @param normalize: whether to normalize trajectories in a way that maximum absolute value along each coordinate is 1 or not
    @param save: whether to save trajectories and energies to trajectories/pendulum or not
    @return data: 3d array containing all created trajectories
    @return energies: energies of each trajectory
    """
    def derivative(arr):
        return np.array([arr[1], -np.sin(arr[0])])

    state0_generator = lambda: np.array([0, np.random.uniform(0.7, 1.3)])

    data = np.array([generate_traj(derivative, state0_generator, energy, "absolute", 0.1, 10, traj_len=traj_len) for _ in tqdm(range(N_traj))])
    energies = np.array([energy(traj[0]) for traj in data])
    data = add_noise(data)

    if normalize:
        data = normalize_data(data)

    if save:
        for trajectory, i in zip(data, range(N_traj)):
            np.savetxt("trajectories/pendulum/" + str(i) + ".csv", trajectory)
        np.savetxt("trajectories/pendulum/energies.csv", energies)

    return data, energies
