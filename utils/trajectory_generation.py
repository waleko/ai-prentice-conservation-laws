import numpy as np

import creating.pendulum as pend
import creating.harmonic_oscillator as ho
import creating.kepler_problem as kp
import creating.double_pendulum as dp
import creating.coupled_oscillator as co


def angle_norm(alpha):
    return ((alpha + np.pi) % (2 * np.pi)) - np.pi


def my_save(data, experiment_name: str):
    N_traj = data.shape[0]
    for trajectory, i in zip(data, range(N_traj)):
        np.savetxt(f"trajectories/{experiment_name}/" + str(i) + ".csv", trajectory)


def generate_all_trajectories(count=200, traj_len=1000):
    pend.create_trajectories(count, traj_len=traj_len, normalize=False)
    ho.create_trajectories(count, traj_len=traj_len, normalize=False)
    kp.create_trajectories(count, traj_len=traj_len, normalize=False)
    dp.create_trajectories(count, traj_len=traj_len, normalize=False)
    co.create_trajectories(count, traj_len=traj_len, normalize=False)

    # double pendulum low/high energies
    dp_low_energies = dp.create_trajectories(count, traj_len, normalize=False, save=False, E_min=-3, E_max=-2)[0]
    dp_high_energies = dp.create_trajectories(count, traj_len, normalize=False, save=False, E_min=0, E_max=0)[0]
    # double pendulum angles to [-pi; pi]
    dp_low_energies[:, :, :2] = np.vectorize(angle_norm)(dp_low_energies[:, :, :2])
    dp_high_energies[:, :, :2] = np.vectorize(angle_norm)(dp_high_energies[:, :, :2])
    # save
    my_save(dp_low_energies, "double_pendulum_low_energy")
    my_save(dp_high_energies, "double_pendulum_high_energy")
