import opengen as og
import casadi.casadi as cs
import numpy as np

# Step 1: Define the ingredients of the problem
(nu, nx, N, L, ts) = (2, 3, 50, 0.5, 0.1)
(xref, yref, thetaref) = (0, 0, 0)
(q, qtheta, r, qN, qthetaN) = (10, 0.1, 1, 200, 20)
(center_x, center_y, radius) = -4, 0, 2

if __name__ == "__main__":
    u = cs.SX.sym('u', nu * N)
    z0 = cs.SX.sym('z0', nx)
    p = cs.SX.sym('p', 1)

    (x, y, theta) = (z0[0], z0[1], z0[2])

    cost = 0
    c = []
    for t in range(0, nu * N, nu):
        cost += q * ((x - xref) ** 2 + (y - yref) ** 2) + qtheta * (theta - thetaref) ** 2
        u_t = u[t:t + 2]
        theta_dot = (1 / L) * (u_t[1] * cs.cos(theta) - u_t[0] * cs.sin(theta))
        cost += r * cs.dot(u_t, u_t)
        x += ts * (u_t[0] + L * cs.sin(theta) * theta_dot)
        y += ts * (u_t[1] - L * cs.cos(theta) * theta_dot)
        theta += ts * theta_dot
        c = cs.vertcat(c, cs.fmax(0, radius ** p - cs.fabs(x - center_x) ** p - cs.fabs(y - center_y) ** p))

    cost += qN * ((x - xref) ** 2 + (y - yref) ** 2) + qthetaN * (theta - thetaref) ** 2

    umin = [-3.0] * (nu * N)
    umax = [3.0] * (nu * N)
    bounds = og.constraints.Rectangle(umin, umax)

    # Step 2: Build your optimizer
    param = cs.vertcat(z0, p)
    problem = og.builder.Problem(u, param, cost).with_constraints(bounds).with_penalty_constraints(c)
    build_config = og.config.BuildConfiguration() \
        .with_build_directory("my_optimizers") \
        .with_build_mode("debug") \
        .with_build_python_bindings()
    meta = og.config.OptimizerMeta() \
        .with_optimizer_name("navigation")
    solver_config = og.config.SolverConfiguration() \
        .with_tolerance(1e-4) \
        .with_initial_tolerance(1e-4) \
        .with_max_outer_iterations(20) \
        .with_delta_tolerance(1e-2) \
        .with_penalty_weight_update_factor(1.5) \
        .with_initial_penalty(100.0)
    builder = og.builder.OpEnOptimizerBuilder(problem,
                                              meta,
                                              build_config,
                                              solver_config)
    builder.build()
