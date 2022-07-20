from obstructed_navigation_solver import nx, nu, L, N, ts
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, 'my_optimizers/navigation')
import navigation
solver = navigation.solver()


def mpc_controller(state):
    mpc_result = solver.run(p=state)
    if mpc_result.exit_status != "Converged":
        print(mpc_result.exit_status)
    return mpc_result.solution[0:nu]


def system_dynamics(state, control):
    x_ = state[0]
    y_ = state[1]
    theta_ = state[2]
    theta_dot_ = (1 / L) * (control[1] * np.cos(theta_) - control[0] * np.sin(theta_))
    return np.array([x_ + ts * (control[0] + L * np.sin(theta_) * theta_dot_),
                     y_ + ts * (control[1] - L * np.cos(theta_) * theta_dot_),
                     theta_ + ts * theta_dot_])


def draw_line(x_coord, y_coord, angle):
    r = .2  # arrow length
    plt.arrow(x_coord, y_coord, r * np.cos(angle), r * np.sin(angle))


initial_state = [-10, 5, np.pi/8]  # [x_coord, y_coord, orientation_angle(+x=0rad,+y=pi/2rad)]
Nsim = 100
x_cache = np.zeros((nx, Nsim))
x_cache[:, 0] = np.array(initial_state)

for t in range(Nsim-1):
    x_current = x_cache[:, t]
    u_mpc = mpc_controller(x_current)
    x_cache[:, t + 1] = system_dynamics(x_current, u_mpc)

# plot x and y points
# plt.plot(x_cache[0], x_cache[1], '-o')
# plt.show()

# plot orientation_angle
# plt.plot(x_cache[2], '-o')
# plt.show()

# plot lines at x and y points at vehicle angle
for i in range(len(x_cache[0])):
    draw_line(x_cache[0][i], x_cache[1][i], x_cache[2][i])

# plot
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.show()
