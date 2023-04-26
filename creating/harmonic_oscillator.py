import numpy as np


def energy(state):
    x, p = state
    return p ** 2 / 2 + x ** 2 / 2


def create_trajectories(N_traj=200, traj_len=200, save=True):
    """
    Creates trajectories of harmonic oscillator with different energies.
    Returns trajectories and energies
    
    @param N_traj: number of created trajectories
    @param traj_len: length of each trajectory
    @param save: whether to save trajectories and energies to trajectories/harmonic_oscillator or not
    
    @return data: 3d array containing all created trajectories
    @return energies: energies of each trajectory
    """
    angles = 2 * np.pi * np.sort(np.random.uniform(size=(N_traj, traj_len)), axis=1)
    radiuses = np.random.uniform(0, 1, size=N_traj)
    
    data = radiuses[:, None, None] * np.stack((np.cos(angles), np.sin(angles)), axis=2)
    energies = radiuses ** 2 / 2

    if save:
        np.savez("../trajectories/harmonic_oscillator.npz", data=data, params=energies[:, None])

    return data, energies
