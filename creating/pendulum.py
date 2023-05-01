import numpy as np
from auxiliary_functions import *


def energy(state):
    theta, L = state
    return L ** 2 / 2 - np.cos(theta) + 1


def create_trajectories(N_traj=200, traj_len=200, save=True):
    """
    Creates trajectories of pendulum with different energies.
    Returns trajectories and energies
    
    @param N_traj: number of created trajectories
    @param traj_len: length of each trajectory
    @param save: whether to save trajectories and energies to trajectories/pendulum or not
    
    @return data: 3d array containing all created trajectories
    @return energies: energies of each trajectory
    """
    def derivative(arr):
        return np.array([arr[1], -np.sin(arr[0])])

    state0_generator = lambda: np.array([0, np.random.uniform(0, 2)])

    data = np.array([generate_traj(derivative, state0_generator, energy, "absolute", 0.1, 10, traj_len=traj_len) for _ in tqdm(range(N_traj))])
    energies = np.array([energy(traj[0]) for traj in data])

    if save:
        np.savez("../trajectories/pendulum.npz", data=data, params=energies[:, None])

    return data, energies
