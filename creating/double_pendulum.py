import numpy as np
import matplotlib.pyplot as plt
from . import auxiliary_functions as af


def single_trajectory(filename, params):
    state0, = params

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

    af.run_and_write(derivative, state0, filename, 0.01, 10000, 1000)


def energy(state):
    theta1, theta2, p1, p2 = state
    denom = 1 + np.sin(theta1 - theta2) ** 2
    return (p1 ** 2 + 2 * p2 ** 2 - 2 * p1 * p2 * np.cos(theta1 - theta2)) / (2 * denom) - 2 * np.cos(theta1) - np.cos(theta2)


def energy12(state, pm):
    theta1, theta2, p1, p2 = state
    denom = 1 + np.sin(theta1 - theta2) ** 2
    omega1, omega2 = (p1 - p2 * np.cos(theta1 - theta2)) / denom, (2 * p2 - p1 * np.cos(theta1 - theta2)) / denom
    return (4 * theta1 ** 2 + 2 * theta2 ** 2 + pm * np.sqrt(2) + (2 + pm * np.sqrt(2)) * (2 * omega1 ** 2 + omega2 ** 2) + 4 * (1 + pm * np.sqrt(2)) * omega1 * omega2) / 8


def create_trajectories(N_traj):
    def single_state():
        E = np.random.uniform(-3, 2)
        K = -1
        while K < 0:
            theta1, theta2 = np.random.uniform(-np.pi / 4, np.pi / 4, size=2)
            K = E + 2 * np.cos(theta1) + np.cos(theta2)
        K *= (1 + np.sin(theta1 - theta2) ** 2)
        p2 = np.sqrt(np.random.uniform(0, K / 2))
        p1 = 2 * p2 * np.cos(theta1 - theta2) + np.sqrt((p2 * np.cos(theta1 - theta2)) ** 2 + K - 2 * p2 ** 2)
        return [theta1, theta2, p1, p2]

    af.create_multiple_trajectories("double_pendulum", N_traj, single_trajectory, zip([single_state() for _ in range(N_traj)]))
    af.add_noise("double_pendulum", N_traj)
    plt.figure()
    for i in range(3):
        traj = af.read_traj("double_pendulum", i)
        x1 = np.sin(traj[:, 0])
        y1 = -np.cos(traj[:, 0])
        x2 = np.sin(traj[:, 1]) + x1
        y2 = -np.cos(traj[:, 1]) + y1
        plt.scatter(np.concatenate((x1, x2)), np.concatenate((y1, y2)), s=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("some double pendulum trajectories")
    plt.savefig("trajectories/double_pendulum/some_trajectories.png")
    af.compute_conserved_quantity("double_pendulum", "E", energy)
    af.compute_conserved_quantity("double_pendulum", "E1", lambda traj: energy12(traj, 1))
    af.compute_conserved_quantity("double_pendulum", "E2", lambda traj: energy12(traj, -1))
    af.normalize("double_pendulum", N_traj)
