import numpy as np
from numpy import cos, sin, sqrt
from tqdm import tqdm
from .auxiliary_functions import *


def energy(state):
    x1, x2, p1, p2 = state
    return x1 ** 2 + x2 ** 2 - x1 * x2 + (p1 ** 2 + p2 ** 2) / 2


def energy1(state):
    x1, x2, p1, p2 = state
    return ((x1 + x2) ** 2 + (p1 + p2) ** 2) / 4


def energy2(state):
    x1, x2, p1, p2 = state
    return (3 * (x1 - x2) ** 2 + (p1 - p2) ** 2) / 4


def single_trajectory(E1, E2, traj_len):
    A = np.sqrt(E1)
    B = np.sqrt(E2) / 2
    t = np.sort(np.random.uniform(0, 100, size=traj_len))
    
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

    return traj


def create_trajectories(N_traj=400, traj_len=500, normalize=True, save=True):
    """
    Creates trajectories of coupled_oscillator with different energies.
    Returns trajectories and energies of two modes
    @param N_traj: number of created trajectories
    @param normalize: whether to normalize trajectories in a way that maximum absolute value along each coordinate is 1 or not
    @param save: whether to save trajectories and energies to trajectories/coupled_oscillator or not
    @return data: 3d array containing all created trajectories
    @return energies1: energies of the first mode for each trajectory
    @return energies2: energies of the second mode for each trajectory
    """
    energies1, energies2 = np.random.uniform(0, 1, size=(2, N_traj))
    data = np.array([single_trajectory(E1, E2, traj_len) for E1, E2 in tqdm(zip(energies1, energies2))])
    data = add_noise(data)
    
    if normalize:
        data = normalize_data(data)

    if save:
        for trajectory, i in zip(data, range(N_traj)):
            np.savetxt("trajectories/coupled_oscillator/" + str(i) + ".csv", trajectory)
        np.savetxt("trajectories/coupled_oscillator/energies1.csv", energies1)
        np.savetxt("trajectories/coupled_oscillator/energies2.csv", energies2)

    return data, energies1, energies2
