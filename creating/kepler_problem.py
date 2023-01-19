import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from numpy.random import uniform
from numpy import cos, sin, pi

from . import auxiliary_functions as af


def single_trajectory(filename: str):
    E, L, theta0 = uniform(-2, -1), uniform(0.25, 0.5), uniform(0, 2 * pi)
    e = np.sqrt(1 + 2 * E * L ** 2)
    random_angles = uniform(0, 2 * pi, size=200)
    r_arr = L ** 2 / (1 + e * np.cos(random_angles))
    x_arr = cos(random_angles) * r_arr
    y_arr = sin(random_angles) * r_arr
    px_arr = -sin(random_angles) / L
    py_arr = (cos(random_angles) + e) / L
    traj = np.stack((x_arr, y_arr, px_arr, py_arr), axis=1)

    def rotate(state):
        c = np.cos(theta0)
        s = np.sin(theta0)
        A = np.array([[c, -s, 0,  0],
                      [s,  c, 0,  0],
                      [0,  0, c, -s],
                      [0,  0, s,  c]])
        return A.dot(state)

    rotate = np.vectorize(rotate, signature="(m)->(m)")

    traj = rotate(traj)

    np.savetxt(filename, traj)


def energy(state):
    r, p = state.reshape(2, 2)
    return (p ** 2).sum() / 2 - 1 / norm(r)


def angular_momentum(state):
    return state[0] * state[3] - state[1] * state[2]


def create_trajectories(N_traj):
    af.create_multiple_trajectories("kepler_problem", N_traj, single_trajectory)
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
    plt.savefig("trajectories/kepler_problem/some_trajectories.png")
    af.compute_conserved_quantity("kepler_problem", "E", energy, N_traj)
    af.compute_conserved_quantity("kepler_problem", "L", angular_momentum)
    af.normalize("kepler_problem", N_traj)
