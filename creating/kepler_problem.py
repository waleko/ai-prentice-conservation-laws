import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from numpy.random import uniform

import auxiliary_functions as af


def create_ellipse(scale, e, theta):
    r = scale / (1 + e * np.cos(theta))
    return np.cos(theta) * r, np.sin(theta) * r


def single_trajectory(filename: str, params: tuple):
    E, L, theta0 = params
    scale = L ** 2
    e = np.sqrt(1 + 2 * E * L ** 2)
    theta = 2 * np.pi * uniform(size=200)
    x, y = create_ellipse(L ** 2, e, theta) 
    px = -L * np.sin(theta) / scale
    py = L * (np.cos(theta) + e) / scale

    def rotate(state, phi):
        c = np.cos(phi)
        s = np.sin(phi)
        A = np.array([[c, -s, 0,  0],
                      [s,  c, 0,  0],
                      [0,  0, c, -s],
                      [0,  0, s,  c]])
        return A.dot(state)

    np.savetxt(filename, [rotate(el, theta0) for el in np.stack([x, y, px, py]).transpose()])


def energy(state):
    return (state[2:] ** 2).sum() / 2 - 1 / norm(state[:2])


def angular_momentum(state):
    return state[0] * state[3] - state[1] * state[2]


def create_trajectories(N_traj):
    E = uniform(-2, -1, size=N_traj)
    L = uniform(0.25, 0.5, size=N_traj)
    params = zip(E, L, 2 * np.pi * uniform(size=N_traj))
    af.create_multiple_trajectories("kepler_problem", N_traj, single_trajectory, params)
    af.add_noise("kepler_problem", N_traj)
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(30, 10)
    for i in range(10):
        traj = af.read_traj("kepler_problem", i)
        axes[0].scatter(traj[:, 0], traj[:, 1], s=1)
        axes[1].scatter(traj[:, 2], traj[:, 3], s=1)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[1].set_xlabel("px")
    axes[1].set_ylabel("py")
    plt.title("some kepler trajectories")
    plt.savefig("../trajectories/kepler_problem/some_trajectories.png")
    af.compute_conserved_quantity("kepler_problem", "E", energy, N_traj)
    af.compute_conserved_quantity("kepler_problem", "L", angular_momentum)
    af.normalize("kepler_problem", N_traj)
