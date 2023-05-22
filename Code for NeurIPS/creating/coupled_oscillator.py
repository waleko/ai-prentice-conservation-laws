import numpy as np
from numpy import cos, sin, sqrt


def energy(state):
    x1, x2, p1, p2 = state
    return x1 ** 2 + x2 ** 2 - x1 * x2 + (p1 ** 2 + p2 ** 2) / 2


def energy1(state):
    x1, x2, p1, p2 = state
    return ((x1 + x2) ** 2 + (p1 + p2) ** 2) / 4


def energy2(state):
    x1, x2, p1, p2 = state
    return (3 * (x1 - x2) ** 2 + (p1 - p2) ** 2) / 4


def create_trajectories(N_traj=200, traj_len=200, save=True):
    """
    Creates trajectories of coupled_oscillator with different energies.
    Returns trajectories and energies of two modes

    @param N_traj: number of created trajectories
    @param traj_len: length of each trajectory
    @param save: whether to save trajectories and energies to trajectories/coupled_oscillator or not

    @return data: 3d array containing all created trajectories
    @return energies1: energies of the first mode for each trajectory
    @return energies2: energies of the second mode for each trajectory
    """
    energies1, energies2 = np.random.uniform(0, 1, size=(2, N_traj))
    A = np.sqrt(energies1)[:, None]
    B = np.sqrt(energies2)[:, None] / 2

    t = np.sort(np.random.uniform(0, 100, size=(N_traj, traj_len)), axis=1)
    
    x1_arr_mode1 = x2_arr_mode1 = cos(t) * A
    p1_arr_mode1 = p2_arr_mode1 = -sin(t) * A


    x1_arr_mode2 = B * cos(sqrt(3) * t)
    x2_arr_mode2 = -B * cos(sqrt(3) * t)
    p1_arr_mode2 = -B * sqrt(3) * sin(sqrt(3) * t)
    p2_arr_mode2 = B * sqrt(3) * sin(sqrt(3) * t)

    x1_arr = x1_arr_mode1 + x1_arr_mode2
    x2_arr = x2_arr_mode1 + x2_arr_mode2
    p1_arr = p1_arr_mode1 + p1_arr_mode2
    p2_arr = p2_arr_mode1 + p2_arr_mode2

    data = np.stack((x1_arr, x2_arr, p1_arr, p2_arr), axis=2)
    
    if save:
        np.savez("../trajectories/coupled_oscillator", data=data, params=np.stack((energies1, energies2), axis=1))

    return data, energies1, energies2
