import numpy as np
import matplotlib.pyplot as plt
from .auxiliary_functions import *


def random_points_on_unit_circle(size):
    random_angles = np.sort(2 * np.pi * np.random.uniform(size=size))
    return np.stack((np.sin(random_angles), np.cos(random_angles)), axis=1)


def energy(state):
    x, p = state
    return p ** 2 / 2 + x ** 2 / 2


def create_trajectories(N_traj=200, traj_len=200, normalize=True, save=True):
    """
    Creates trajectories of harmonic oscillator with different energies. Returns trajectories and energies
    @param N_traj: number of created trajectories
    @param normalize: whether to normalize trajectories in a way that maximum absolute value along each coordinate is 1 or not
    @param save: whether to save trajectories and energies to trajectories/harmonic_oscillator or not
    @return data: 3d array containing all created trajectories
    @return energies: energies of each trajectory
    """
    N_circles = np.array([random_points_on_unit_circle(size=traj_len) for _ in tqdm(range(N_traj))])
    r_arr = np.random.uniform(0, 1, size=(N_traj, 1, 1))
    data = N_circles * r_arr
    energies = r_arr.reshape(N_traj) ** 2 / 2
    data = add_noise(data)
    
    if normalize:
        data = normalize_data(data)

    if save:
        for trajectory, i in zip(data, range(N_traj)):
            np.savetxt("trajectories/harmonic_oscillator/" + str(i) + ".csv", trajectory)
        np.savetxt("trajectories/harmonic_oscillator/energies.csv", energies)

    return data, energies
