import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from time import time

from typing import Callable


def fourth_order_runge_kutta(derivative: Callable[[np.ndarray], np.ndarray],
                              state0: np.ndarray,
                              conserved_quantity: Callable[[np.ndarray], float],
                              deviation_type: str,
                              T: float,
                              dt_max: float=0.1,
                              k: float=1.2,
                              max_deviation_threshold: float=0.01,
                              min_deviation_threshold: float=0.0001) -> list:
    def next_state(cur_state, dt_):
        k1 = derivative(cur_state)
        k2 = derivative(cur_state + k1 * dt_ / 2)
        k3 = derivative(cur_state + k2 * dt_ / 2)
        k4 = derivative(cur_state + k3 * dt_)
        return cur_state + (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6) * dt
    
    dt = dt_max
    states = [state0]
    dt_arr = []
    start_time = time()
    while True:
        while True:
            if time() - start_time > 5:
                return None, None
            deviation_ = deviation(conserved_quantity(states[-1]), conserved_quantity(next_state(states[-1], dt)), deviation_type)
            if deviation_ > max_deviation_threshold:
                dt /= k
            elif deviation_ < min_deviation_threshold and dt < dt_max:
                dt *= k
            else:
                break
        T -= dt
        states.append(next_state(states[-1], dt))
        dt_arr.append(dt)
        if T < 0:
            return states, dt_arr


def run_and_write(derivative: Callable[[np.ndarray], np.ndarray],
                   state0_generator: np.ndarray,
                   filename: str,
                   conserved_quantity: Callable[[np.ndarray], float],
                   deviation_type: str,
                   dt: float,
                   T: float,
                   traj_size: int=200,
                   max_deviation_threshold: float=0.01,
                   min_deviation_threshold: float=0.0001):
    traj, dt_arr = fourth_order_runge_kutta(derivative,
                                             state0_generator(),
                                             conserved_quantity,
                                             deviation_type,
                                             T,
                                             dt,
                                             max_deviation_threshold=max_deviation_threshold,
                                             min_deviation_threshold=min_deviation_threshold)
    while traj is None:
        traj, dt_arr = fourth_order_runge_kutta(derivative,
                                                state0_generator(),
                                                conserved_quantity,
                                                deviation_type,
                                                T,
                                                dt,
                                                max_deviation_threshold=max_deviation_threshold,
                                                min_deviation_threshold=min_deviation_threshold)
    traj = np.array(traj)
    i = np.random.choice(np.arange(len(traj) - 1), p=dt_arr / np.sum(dt_arr), size=traj_size)
    w = np.random.uniform(size=(traj_size, 1))
    np.savetxt(filename, traj[i] * w + traj[i + 1] * (1 - w))

    
def create_multiple_trajectories(name: str, N_trajectories: int, create_single_trajectory: Callable[[str], None]):
    print("\nCreating trajectories for " + name)
    for i in tqdm(range(N_trajectories)):
        create_single_trajectory("trajectories/" + name + "/" + str(i) + ".csv")


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


def deviation(prev_value, new_value, deviation_type):
    if deviation_type == "absolute":
        return abs(prev_value - new_value)
    if deviation_type == "relative":
        return abs(1 - new_value / prev_value)
