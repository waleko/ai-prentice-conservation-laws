import numpy as np


def choose_trajectories_low_and_high_energies(N_traj_le=200, N_traj_he=200, le_boundary=-2, he_boundary=-1):
    double_pendulum_data = np.load("../trajectories/double_pendulum_2000_500_x1.5pi_v0.5.npz")
    
    data = double_pendulum_data["data"]
    params = double_pendulum_data["params"]
    energies = params[:, 0]

    if (energies < le_boundary).sum() < N_traj_le:
        print("Not enough trajectories with low energy")
    else:
        le_ind = np.random.choice(np.where(energies < le_boundary)[0], size=N_traj_le, replace=False)
        data_le = data[le_ind]
        params_le = params[le_ind]
        np.savez("../trajectories/double_pendulum_low_energy.npz", data=data_le, params=params_le)

    if (energies > he_boundary).sum() < N_traj_he:
        print("Not enough trajectories with high energy")
    else:
        he_ind = np.random.choice(np.where(energies > he_boundary)[0], size=N_traj_he, replace=False)
        data_he = data[he_ind]
        params_he = params[he_ind]
        np.savez("../trajectories/double_pendulum_high_energy.npz", data=data_he, params=params_he)
