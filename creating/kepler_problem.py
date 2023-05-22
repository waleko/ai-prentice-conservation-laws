import numpy as np
from scipy.linalg import norm
from numpy.random import uniform
from numpy import cos, sin, pi
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid


def single_trajectory(E, L, theta0, traj_len):
    e = np.sqrt(1 + 2 * E * L ** 2)
    
    phi_arr = np.linspace(0, 2 * pi, 10000)
    t_arr = cumulative_trapezoid((1 + e * cos(phi_arr)) **  (-2), phi_arr)
    angles = phi_arr[np.searchsorted(t_arr, uniform(0, t_arr[-1], size=traj_len))]

    r_arr = L ** 2 / (1 + e * cos(angles))
    x_arr = cos(angles) * r_arr
    y_arr = sin(angles) * r_arr
    px_arr = -sin(angles) / L
    py_arr = (cos(angles) + e) / L
    traj = np.stack((x_arr, y_arr, px_arr, py_arr), axis=1)

    def rotate(state):
        c = cos(theta0)
        s = sin(theta0)
        A = np.array([[c, -s, 0,  0],
                      [s,  c, 0,  0],
                      [0,  0, c, -s],
                      [0,  0, s,  c]])
        return A.dot(state)

    traj = np.vectorize(rotate, signature="(m)->(m)")(traj)

    return traj


def energy(state):
    r, p = state.reshape(2, 2)
    return (p ** 2).sum() / 2 - 1 / norm(r)


def angular_momentum(state):
    return state[0] * state[3] - state[1] * state[2]


def create_trajectories(N_traj=200, traj_len=500, save=True, name="kepler_problem", energy_interval=(-0.5, -0.15), momentum_interval=(0, 1), orientation_interval=(0, pi / 4)):
    """
    Creates trajectories of kepler problem with different energies.
    Returns trajectories, energies, angular momentums and directions of Runge-Lenz vector

    @param N_traj: number of created trajectories
    @param traj_len: length of each trajectory
    @param save: whether to save trajectories and conserved quantities to trajectories/kepler_problem or not
    @param name: name of the file where to save the results
    @param energy_interval: from that interval energies are sampled
    @param momentum_interval: from that interval angular momentums are sampled
    @param orientation_interval: from that interval directions of Runge-Lenz vectors are sampled

    @return data: 3d array containing all created trajectories
    @return energies: energies of each trajectory
    @return angular_momentums: angular momentums of each trajectory
    @return thetas0: directions of the Runge-Lenz vector of each trajectory
    """
    energies = uniform(*energy_interval, size=N_traj)
    angular_momentums = uniform(*momentum_interval, size=N_traj)
    thetas0 = uniform(*orientation_interval, size=N_traj)
    
    data = np.array([single_trajectory(E, L, theta0, traj_len) for E, L, theta0 in tqdm(np.stack((energies, angular_momentums, thetas0), axis=1))])

    if save:
        np.savez(f"../trajectories/{name}.npz", data=data, params=np.stack((energies, angular_momentums, thetas0), axis=1))

    return data, energies, angular_momentums, thetas0
