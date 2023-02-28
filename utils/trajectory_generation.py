import creating.pendulum as pend
import creating.harmonic_oscillator as ho
import creating.kepler_problem as kp
import creating.double_pendulum as dp
import creating.coupled_oscillator as co


def generate_all_trajectories(count=200, traj_len=1000):
    pend.create_trajectories(count, traj_len=traj_len, normalize=False)
    ho.create_trajectories(count, traj_len=traj_len, normalize=False)
    kp.create_trajectories(count, traj_len=traj_len, normalize=False)
    dp.create_trajectories(count, traj_len=traj_len, normalize=False)
    co.create_trajectories(count, traj_len=traj_len, normalize=False)
