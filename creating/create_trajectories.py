import pendulum as pend
import harmonic_oscillator as ho
import kepler_problem as kp
import double_pendulum as dp
import three_body_problem as tbp
import coupled_oscillator as co


pend.create_trajectories(4000)
ho.create_trajectories(4000)
kp.create_trajectories(4000)
dp.create_trajectories(1000)
# tbp.create_trajectories(200)
co.create_trajectories(4000)
