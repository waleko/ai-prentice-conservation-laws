import creating.pendulum as pend
import creating.harmonic_oscillator as ho
import creating.kepler_problem as kp
import creating.coupled_oscillator as co


def generate_all_trajectories(count=200, traj_len=1000):
    pend.create_trajectories(count, traj_len=traj_len, save=True)
    ho.create_trajectories(count, traj_len=traj_len, save=True)
    kp.create_trajectories(count, traj_len=traj_len, save=True)
    co.create_trajectories(count, traj_len=traj_len, save=True)
