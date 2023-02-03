import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from time import time

from typing import Callable, Literal, Union, Tuple, List


def fourth_order_runge_kutta(derivative: Callable[[np.ndarray], np.ndarray], state0: np.ndarray,
                             conserved_quantity: Callable[[np.ndarray], float], deviation_type: str,
                             dt_max: float, T: float, k: float=1.2, max_deviation_threshold: float=0.01,
                             min_deviation_threshold: float=0.0001) -> Union[Tuple[List[np.ndarray], List[float]], None]:
    """
    Generates uniformli distributed (with respect to the time) points on the trajectory
    @param derivative: derivetive of the state with respect to time as a function of the state
    @param state0: start state
    @param conserved_quantity: conserved quantity as a function of the state, which is used to control accuracy of the Runge-Kutta algorithm
    @param deviation_type: type of the deviation of the conserved quantity, which is used to  control accuracy of the Runge-Kutta algorithm
    @param dt_max: max time step for Runge-Kutta algorithm
    @param T: time for the system to evolve
    @param k: parameter, which controls strength of time step adjustment
    @param max_deviation_threshold: maximal threshold for the deviation of the conserved quantity
    @param min_deviation_threshold: minimal threshold for the deviation of the conserved quantity
    @return states: trajectory points created by Runge-Kutta algorithm
    @return dt_arr: list of time steps made by Runge-Kutta algorithm
    """

    # Runge-Kutta step
    def next_state(cur_state, dt_):
        k1 = derivative(cur_state)
        k2 = derivative(cur_state + k1 * dt_ / 2)
        k3 = derivative(cur_state + k2 * dt_ / 2)
        k4 = derivative(cur_state + k3 * dt_)
        return cur_state + (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6) * dt
    
    # initializing
    dt = dt_max
    states = [state0]
    dt_arr = []
    start_time = time()

    # running Runge-Kutta algorithm    
    while True:

        # adjusting time step according to the conservation deviation
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
        
        # making step
        T -= dt
        states.append(next_state(states[-1], dt))
        dt_arr.append(dt)
        if T < 0:
            return states, dt_arr


def generate_traj(derivative: Callable[[np.ndarray], np.ndarray], state0_generator: np.ndarray,
                  conserved_quantity: Callable[[np.ndarray], float], deviation_type: Literal["absolute", "relative"], dt_max: float, T: float,
                  traj_len: int=200, max_deviation_threshold: float=0.01, min_deviation_threshold: float=0.0001):
    """
    Generates uniformli distributed (with respect to the time) points on the trajectory
    @param derivative: derivetive of the state with respect to time as a function of the state
    @param state0_generator: function, which generates random start state
    @param conserved_quantity: conserved quantity as a function of the state, which is used to control accuracy of the Runge-Kutta algorithm
    @param deviation_type: type of the deviation of the conserved quantity, which is used to  control accuracy of the Runge-Kutta algorithm
    @param dt_max: max time step for Runge-Kutta algorithm
    @param T: time for the system to evolve
    @param traj_len: number of points on the trajectory to generate
    @param max_deviation_threshold: maximal threshold for the deviation of the conserved quantity
    @param min_deviation_threshold: minimal threshold for the deviation of the conserved quantity
    @return traj_points: uniformly distributed on the trajectory ordered by the time
    """

    # finding the trajectory by Runge-Kutta algorithm
    traj, dt_arr = None, None
    while traj is None:
        traj, dt_arr = fourth_order_runge_kutta(derivative, state0_generator(), conserved_quantity, deviation_type, dt_max, T,
                                                max_deviation_threshold=max_deviation_threshold, min_deviation_threshold=min_deviation_threshold)
    traj = np.array(traj)

    # choosing random indices to choose random intervals
    i = np.random.choice(np.arange(len(traj) - 1), p=dt_arr / np.sum(dt_arr), size=traj_len)
    # choosing the weights for weighted sum of the ends of the interval to choose random points from the intervals
    w = np.random.uniform(size=traj_len)
    # sorting indices and weights first by indices and then by weights to order all points by time
    i, w = np.array(sorted(np.stack((i, w), axis=1), key=lambda x: x[0] + x[1] / 10)).transpose()
    i = i.astype(int)
    w = w.reshape((traj_len, 1))

    traj_points = (1 - w) * traj[i] + w * traj[i + 1]
    return traj_points
  

def deviation(prev_value, new_value, deviation_type):
    if deviation_type == "absolute":
        return abs(prev_value - new_value)
    if deviation_type == "relative":
        return abs(1 - new_value / prev_value)


def add_noise(data, strength=0.01):
    stds = np.std(data, axis=1)
    scales = np.stack([stds] * data.shape[1], axis=1) * strength
    noise = np.random.normal(scale=scales)
    return data + noise


def normalize_data(data):
    return data / np.abs(data).max(axis=(0, 1)).reshape(1, 1, data.shape[2])
