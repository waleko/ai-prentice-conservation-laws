import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from typing import Callable


def fourth_order_runge_kutta(derivative: Callable[[np.ndarray], np.ndarray], state0: np.ndarray, dt: float=0.01, N_steps: int=1000) -> list:
    states = [state0]
    for _ in range(N_steps):
        k1 = derivative(states[-1])
        k2 = derivative(states[-1] + k1 * dt / 2)
        k3 = derivative(states[-1] + k2 * dt / 2)
        k4 = derivative(states[-1] + k3 * dt)
        states.append(states[-1] + (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6) * dt)
        if np.allclose(state0, states[-1], atol=dt) and (not np.allclose(state0, states[-2], atol=dt)):
            break
    return np.array(states)


def run_and_write(derivative: Callable[[np.ndarray], np.ndarray], state0: np.ndarray, filename: str, dt: float, N_steps: int, max_traj_len: int=200):
    traj = fourth_order_runge_kutta(derivative, state0, dt, N_steps)
    if len(traj) > max_traj_len:
        traj = traj[np.random.choice(np.arange(len(traj)), replace=False, size=max_traj_len)]
    else:
        traj = traj[np.random.choice(np.arange(len(traj)), replace=True, size=max_traj_len)]

    np.savetxt(filename, traj)


def create_multiple_trajectories(name: str, N_trajectories: int, create_single_trajectory: Callable[[str, tuple], None], params: list):
    print("\nCreating trajectories for " + name)
    for i, params_tuple in tqdm(zip(range(N_trajectories), params)):
        create_single_trajectory("trajectories/" + name + "/" + str(i) + ".csv", params_tuple)


def read_traj(name, i):
    return np.genfromtxt("trajectories/" + name + "/" + str(i) + ".csv")


def compute_conserved_quantity(model_name, quantity_name, quantity, N_traj=200):
    values = np.array([np.vectorize(quantity, signature="(m)->()")(read_traj(model_name, i)).mean() for i in range(N_traj)])
    np.savetxt("trajectories/" + model_name + "/" + quantity_name + ".csv", values)


def normalize(name, N_traj):
    data = np.array([read_traj(name, i) for i in range(N_traj)])
    for traj, i in zip(data / np.abs(data).max(axis=(0, 1)).reshape(1, 1, data.shape[2]), range(N_traj)):
        np.savetxt("trajectories/" + name + "/" + str(i) + ".csv", traj)


def add_noise(name, N_traj, strength=0.01):
    data = np.array([read_traj(name, i) for i in range(N_traj)])
    
    def add_noise_to_series(series):
        scale = np.sqrt(((series - series.mean()) ** 2).sum() / (len(series) - 1)) * strength
        return series + np.random.normal(loc=0, scale=scale, size=len(series))

    for traj, i in zip(data, range(N_traj)):
        np.savetxt("trajectories/" + name + "/" + str(i) + ".csv", np.array([add_noise_to_series(coord) for coord in traj.transpose()]).transpose())
