import numpy as np
import matplotlib.pyplot as plt
import auxiliary_functions as af


def single_trajectory(filename: str, params: tuple):
    state0, = params

    def derivative(arr):
        return np.array([arr[1], -np.sin(arr[0])])
    
    af.run_and_write(derivative, state0, filename, 0.01, 1000)


def state0_arr(N_traj):
    return np.stack([np.zeros(N_traj), np.sqrt(np.random.uniform(0.1, 3, size=N_traj))]).transpose()


def energy(state):
    return state[1] ** 2 / 2 - np.cos(state[0])


def create_trajectories(N_traj=200):
    af.create_multiple_trajectories("pendulum", N_traj, single_trajectory, zip(state0_arr(N_traj)))
    af.add_noise("pendulum", N_traj)
    plt.figure()
    for i in range(10):
        traj = af.read_traj("pendulum", i)
        plt.scatter(traj[:, 0], traj[:, 1], s=1)
    plt.xlabel("theta")
    plt.ylabel("L")
    plt.title("some pendulum trajectories")
    plt.savefig("../trajectories/pendulum/some_trajectories.png")
    af.compute_conserved_quantity("pendulum", "E", energy, N_traj)
    af.normalize("pendulum", N_traj)
