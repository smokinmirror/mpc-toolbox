from obstructed_navigation_solver import nx, nu, L, ts, center_x, center_y, radius
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, 'my_optimizers/navigation')
import navigation
solver = navigation.solver()
p_min = 2
p_max = 10


def mpc_controller(state):
    mpc_result = None
    for p_runup in np.arange(p_min, p_max, (p_max-p_min)/4):
        if mpc_result is not None:
            mpc_result = solver.run(p=np.hstack((state, p_runup)),
                                    initial_guess=mpc_result.solution)
        else:
            mpc_result = solver.run(p=np.hstack((state, p_runup)))

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
    length_ = .2  # arrow length
    plt.arrow(x_coord, y_coord, length_ * np.cos(angle), length_ * np.sin(angle))


def draw_circle(center_x_, center_y_, radius_):
    h_ = 0.1  # step length between points
    t_ = np.arange(0, 2*np.pi+h_, h_)  # points
    # circle parameters
    s_x = 1  # scaling on x axis
    s_y = 1  # scaling on y axis
    # create circle points
    x = center_x_ + (radius_ * np.sign(np.sin(t_)) * abs(np.sin(t_))**(2/p_max) * s_x)
    y = center_y_ + (radius_ * np.sign(np.cos(t_)) * abs(np.cos(t_))**(2/p_max) * s_y)
    # rotation parameters
    theta_ = 0  # radians // rotation angle
    # rotate circle points
    rotated_x = np.cos(theta_) * x + np.sin(theta_) * y
    rotated_y = -np.sin(theta_) * x + np.cos(theta_) * y
    # plot points
    for i_ in range(len(t_)):
        plt.plot(rotated_x, rotated_y)


initial_state = [-10, 5, np.pi/4]  # [x_coord, y_coord, orientation_angle(+x=0rad,+y=pi/2rad)]
Nsim = 100
x_cache = np.zeros((nx, Nsim))
x_cache[:, 0] = np.array(initial_state)

for t in range(Nsim-1):
    x_current = x_cache[:, t]
    u_mpc = mpc_controller(x_current)
    x_cache[:, t + 1] = system_dynamics(x_current, u_mpc)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')

# # plot x and y points
# plt.plot(x_cache[0], x_cache[1], '-o')
# plt.show()
#
# # plot x against time
# plt.plot(x_cache[0], '-o')
# plt.show()
#
# # plot y against time
# plt.plot(x_cache[1], '-o')
# plt.show()

# plot orientation_angle
# plt.plot(x_cache[2], '-o')
# plt.show()

# plot lines at x and y points at vehicle angle
for i in range(len(x_cache[0])):
    draw_line(x_cache[0][i], x_cache[1][i], x_cache[2][i])

# plot obstacle
draw_circle(center_x, center_y, radius)

# plot
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.show()
