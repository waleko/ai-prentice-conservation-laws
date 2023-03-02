import numpy as np
from .auxiliary_functions import *


def energy(state):
    theta1, theta2, p1, p2 = state
    denom = 1 + np.sin(theta1 - theta2) ** 2
    return (p1 ** 2 + 2 * p2 ** 2 - 2 * p1 * p2 * np.cos(theta1 - theta2)) / (2 * denom) - 2 * np.cos(theta1) - np.cos(theta2)


def energy12(state, pm):
    theta1, theta2, p1, p2 = state
    denom = 1 + np.sin(theta1 - theta2) ** 2
    omega1, omega2 = (p1 - p2 * np.cos(theta1 - theta2)) / denom, (2 * p2 - p1 * np.cos(theta1 - theta2)) / denom
    return (4 * theta1 ** 2 + 2 * theta2 ** 2 + pm * np.sqrt(2) + (2 + pm * np.sqrt(2)) * (2 * omega1 ** 2 + omega2 ** 2) + 4 * (1 + pm * np.sqrt(2)) * omega1 * omega2) / 8


energy1 = lambda state: energy12(state, 1)
energy2 = lambda state: energy12(state, -1)


def create_trajectories(N_traj=1000, traj_len=500, normalize=True, save=True):
    """
    Creates trajectories of double pendulum with different energies.
    Returns trajectories, energies and energies of two modes
    @param N_traj: number of created trajectories
    @param normalize: whether to normalize trajectories in a way that maximum absolute value along each coordinate is 1 or not
    @param save: whether to save trajectories and energies to trajectories/double_pendulum or not
    @return data: 3d array containing all created trajectories
    @return energies: energies of each trajectory
    @return energies1: energies of the first mode for each trajectory
    @return energies2: energies of the second mode for each trajectory
    """
    def derivative(state):
        theta1, theta2, ptheta1, ptheta2 = state
        denom = 1 + np.sin(theta1 - theta2) ** 2

        theta1dot = (ptheta1 - ptheta2 * np.cos(theta1 - theta2)) / denom
        theta2dot = (2 * ptheta2 - ptheta1 * np.cos(theta1 - theta2)) / denom

        h1 = ptheta1 * ptheta2 * np.sin(theta1 - theta2) / denom
        h2 = (ptheta1 ** 2 + 2 * ptheta2 ** 2 - 2 * ptheta1 * ptheta2 * np.cos(theta1 - theta2)) / (2 * denom ** 2)
        h = h1 - h2 * np.sin(2 * (theta1 - theta2))

        ptheta1dot = -2 * np.sin(theta1) - h
        ptheta2dot = -np.sin(theta2) + h

        return np.array([theta1dot, theta2dot, ptheta1dot, ptheta2dot])

    def state0_generator():
        total_energy = np.random.uniform(-3, 0)
        kinetic_energy = -1
        while kinetic_energy < 0:
            theta1, theta2 = np.random.uniform(-np.pi / 4, np.pi / 4, size=2)
            kinetic_energy = total_energy + 2 * np.cos(theta1) + np.cos(theta2)
        kinetic_energy *= (1 + np.sin(theta1 - theta2) ** 2) * 2
        p2 = np.sqrt(np.random.uniform(0, kinetic_energy / 2))
        p1 = p2 * np.cos(theta1 - theta2) + np.sqrt((p2 * np.cos(theta1 - theta2)) ** 2 + kinetic_energy - 2 * p2 ** 2)
        return [theta1, theta2, p1, p2]
    
    data = np.array([generate_traj(derivative, state0_generator, energy, "absolute", 0.1, 100, max_deviation_threshold=0.1, traj_len=traj_len) for _ in tqdm(range(N_traj))])
    energies = np.array([energy(traj[0]) for traj in data])
    energies1 = np.array([energy1(traj[0]) for traj in data])
    energies2 = np.array([energy2(traj[0]) for traj in data])
    data = add_noise(data)

    if normalize:
        data = normalize_data(data)

    if save:
        for trajectory, i in zip(data, range(N_traj)):
            np.savetxt("trajectories/double_pendulum/" + str(i) + ".csv", trajectory)
        np.savetxt("trajectories/double_pendulum/energies.csv", energies)
        np.savetxt("trajectories/double_pendulum/energies1.csv", energies1)
        np.savetxt("trajectories/double_pendulum/energies2.csv", energies2)

    return data, energies, energies1, energies2