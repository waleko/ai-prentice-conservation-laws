import numpy as np
from scipy.linalg import norm
from numpy.random import uniform
from numpy import cos, sin, pi
from tqdm import tqdm
from scipy.integrate import quad

from .auxiliary_functions import *


def single_trajectory(E, L, theta0, size):
    e = np.sqrt(1 + 2 * E * L ** 2)
    def time(psi):
        return quad(lambda phi: (1 + e * cos(phi)) **  (-2), 0, psi)[0]
    whole_time = time(2 * pi)
    random_time_arr = uniform(0, whole_time, size=size)
    def binary_search(t):
        left = 0.0
        right = 2 * pi
        while right - left > 1e-03:
            mid = (left + right) / 2
            if time(mid) > t:
                right = mid
            else:
                left = mid
        return left
    random_angles_arr = np.vectorize(binary_search)(random_time_arr)
    r_arr = L ** 2 / (1 + e * cos(random_angles_arr))
    x_arr = cos(random_angles_arr) * r_arr
    y_arr = sin(random_angles_arr) * r_arr
    px_arr = -sin(random_angles_arr) / L
    py_arr = (cos(random_angles_arr) + e) / L
    traj = np.stack((x_arr, y_arr, px_arr, py_arr), axis=1)

    def rotate(state):
        c = cos(theta0)
        s = sin(theta0)
        A = np.array([[c, -s, 0,  0],
                      [s,  c, 0,  0],
                      [0,  0, c, -s],
                      [0,  0, s,  c]])
        return A.dot(state)

    rotate = np.vectorize(rotate, signature="(m)->(m)")

    traj = rotate(traj)

    return traj


def energy(state):
    r, p = state.reshape(2, 2)
    return (p ** 2).sum() / 2 - 1 / norm(r)


def angular_momentum(state):
    return state[0] * state[3] - state[1] * state[2]


def create_trajectories(N_traj=400, traj_len=200, normalize=True, save=True):
    """
    Creates trajectories of kepler problem with different energies.
    Returns trajectories, energies, angular momentums and directions of Runge-Lenz vector
    @param N_traj: number of created trajectories
    @param normalize: whether to normalize trajectories in a way that maximum absolute value along each coordinate is 1 or not
    @param save: whether to save trajectories and conserved quantities to trajectories/kepler_problem or not
    @return data: 3d array containing all created trajectories
    @return energies: energies of each trajectory
    @return angular_momentums: angular momentums of each trajectory
    @return thetas0: directions of the Runge-Lenz vector of each trajectory
    """
    energies, angular_momentums, thetas0 = uniform(-0.5, -0.15, size=N_traj), uniform(0, 1, size=N_traj), uniform(0, 2 * pi, size=N_traj)
    
    data = np.array([single_trajectory(E, L, theta0, traj_len) for E, L, theta0 in tqdm(np.stack((energies, angular_momentums, thetas0), axis=1))])
    data = add_noise(data)

    if normalize:
        data = normalize_data(data)

    if save:
        for trajectory, i in zip(data, range(N_traj)):
            np.savetxt("trajectories/kepler_problem/" + str(i) + ".csv", trajectory)
        np.savetxt("trajectories/kepler_problem/energies.csv", energies)
        np.savetxt("trajectories/kepler_problem/angular_momentums.csv", angular_momentums)
        np.savetxt("trajectories/kepler_problem/runge-lenz-direction.csv", thetas0)

    return data, energies, angular_momentums, thetas0
