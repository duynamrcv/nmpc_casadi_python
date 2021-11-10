import casadi as ca
import numpy as np
import time

from draw import Draw_MPC_Obstacle

def shift(T, t0, x0, u, x_n, f):
    f_value = f(x0, u[0])
    st = x0 + T*f_value
    t = t0 + T
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    return t, st, u_end, x_n

def predict_state(x0, u, T, N):
    # Parameter
    d = 0.2
    M = 10; J = 2
    Bx = 0.05; By = 0.05; Bw = 0.06
    # Cx = 0.5; Cy = 0.5; Cw = 0.6
    # define prediction horizon function
    states = np.zeros((N+1, 6))
    states[0,:] = x0
    # euler method
    for i in range(N):
        states[i+1, 0] = states[i, 0] + (states[i, 3]*np.cos(states[i, 2]) - states[i, 4]*np.sin(states[i, 2]))*T
        states[i+1, 1] = states[i, 1] + (states[i, 3]*np.sin(states[i, 2]) + states[i, 4]*np.cos(states[i, 2]))*T
        states[i+1, 2] = states[i, 2] + states[i, 5]*T
        states[i+1, 3] = (0*u[i, 0] + np.sqrt(3)/2*u[i, 1] - np.sqrt(3)/2*u[i, 2] - Bx*states[3])/M
        states[i+1, 4] = (-1*u[i, 0] + 1/2*u[i, 1] + 1/2*u[i, 2] - By*states[4])/M
        states[i+1, 5] = (d*u[i, 0] + d*u[i, 1] + d*u[i, 2] - Bw*states[5])/J
    return states

if __name__ == "__main__":
    # Parameter
    d = 0.2
    M = 10; J = 2
    Bx = 0.05; By = 0.05; Bw = 0.06
    # Cx = 0.5; Cy = 0.5; Cw = 0.6

    T = 0.2                 # time step
    N = 30                  # horizon length
    rob_diam = 0.3          # [m]
    v_max = 1.0             # linear velocity max
    omega_max = np.pi/3.0   # angular velocity max
    u_max = 5               # force max of each direction

    opti = ca.Opti()
    # control variables, toruqe of each wheel
    opt_controls = opti.variable(N, 3)
    u1 = opt_controls[:, 0]
    u2 = opt_controls[:, 1]
    u3 = opt_controls[:, 2]
    
    # state variable: position and velocity
    opt_states = opti.variable(N+1, 6)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]
    vx = opt_states[:, 3]
    vy = opt_states[:, 4]
    omega = opt_states[:, 5]

    # parameters
    opt_x0 = opti.parameter(6)
    opt_xs = opti.parameter(6)

    # create model
    f = lambda x_, u_: ca.vertcat(*[x_[3]*ca.cos(x_[2]) - x_[4]*ca.sin(x_[2]), 
                                    x_[3]*ca.sin(x_[2]) + x_[4]*ca.cos(x_[2]),
                                    x_[5],
                                    (0*u_[0] +  ca.sqrt(3)/2*u_[1] - ca.sqrt(3)/2*u_[2] - Bx*x_[3])/M,
                                    (-1*u_[0] + 1/2*u_[1] + 1/2*u_[2] - By*x_[4])/M,
                                    (d*u_[0] +  d*u_[1] + d*u_[2] - Bw*x_[5])/J])
    f_np = lambda x_, u_: np.array([x_[3]*ca.cos(x_[2]) - x_[4]*ca.sin(x_[2]), 
                                    x_[3]*ca.sin(x_[2]) + x_[4]*ca.cos(x_[2]),
                                    x_[5],
                                    (0*u_[0] +  ca.sqrt(3)/2*u_[1] - ca.sqrt(3)/2*u_[2] - Bx*x_[3])/M,
                                    (-1*u_[0] + 1/2*u_[1] + 1/2*u_[2] - By*x_[4])/M,
                                    (d*u_[0] +  d*u_[1] + d*u_[2] - Bw*x_[5])/J])

    # initial condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :] == x_next)

    # obstacle
    obs_x = [1.5, 0.5, 3.0, 3.5, 4.5]
    obs_y = [1.0, 1.5, 2.7, 4.0, 4.0]
    obs_diam = 0.5
    bias = 0.02

    # add constraints to obstacle
    for i in range(N+1):
        for j in range(len(obs_x)):
            temp_constraints_ = ca.sqrt((opt_states[i, 0]-obs_x[j]-bias)**2+(opt_states[i, 1]-obs_y[j])**2-bias)-rob_diam/2.0-obs_diam/2.0
            opti.subject_to(opti.bounded(0.0, temp_constraints_, 10.0))
        

    # weight matrix
    Q = np.diag([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    R = np.diag([0.01, 0.01, 0.01])

    # cost function
    obj = 0
    for i in range(N):
        obj = obj + ca.mtimes([(opt_states[i, :] - opt_xs.T), Q, (opt_states[i, :] - opt_xs.T).T]) \
                    + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
    opti.minimize(obj)

    # boundary and control conditions
    opti.subject_to(opti.bounded(-10.0, x, 10.0))
    opti.subject_to(opti.bounded(-10.0, y, 10.0))
    opti.subject_to(opti.bounded(-v_max, vx, v_max))
    opti.subject_to(opti.bounded(-v_max, vy, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))
    opti.subject_to(opti.bounded(-u_max, u1, u_max))
    opti.subject_to(opti.bounded(-u_max, u2, u_max))
    opti.subject_to(opti.bounded(-u_max, u3, u_max))

    opts_setting = {'ipopt.max_iter': 100,
                    'ipopt.print_level': 0,
                    'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}
    opti.solver('ipopt', opts_setting)

    # The final state
    final_state = np.array([4.5, 4.5, 0.0, 0.0, 0.0, 0.0])
    opti.set_value(opt_xs, final_state)

    # The initial state
    t0 = 0
    init_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u0 = np.zeros((N, 3))
    current_state = init_state.copy()
    next_states = np.zeros((N+1, 6))
    x_c = []    # contains for the history of the state
    u_c = []
    t_c = [t0]  # for the time
    xx = []
    sim_time = 30.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    while(np.linalg.norm(current_state - final_state) > 1e-2 and mpciter - sim_time/T < 0.0  ):
        # set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x0, current_state)

        # set optimizing target withe init guess
        opti.set_initial(opt_controls, u0)# (N, 3)
        opti.set_initial(opt_states, next_states) # (N+1, 6)
        
        # solve the problem once again
        t_ = time.time()
        sol = opti.solve()
        index_t.append(time.time()- t_)
        # opti.set_initial(opti.lam_g, sol.value(opti.lam_g))
        
        # obtain the control input
        u_res = sol.value(opt_controls)
        u_c.append(u_res[0, :])
        t_c.append(t0)
        next_states_pred = sol.value(opt_states)# prediction_state(x0=current_state, u=u_res, N=N, T=T)
        
        # next_states_pred = prediction_state(x0=current_state, u=u_res, N=N, T=T)
        x_c.append(next_states_pred)
        
        # for next loop
        t0, current_state, u0, next_states = shift(T, t0, current_state, u_res, next_states_pred, f_np)
        mpciter = mpciter + 1
        xx.append(current_state)

    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time)/(mpciter))

    # after loop
    print(mpciter)
    print('final error {}'.format(np.linalg.norm(final_state-current_state)))
    
    # draw function
    draw_result = Draw_MPC_Obstacle(rob_diam=rob_diam,
                                    init_state=init_state,
                                    target_state=final_state,
                                    robot_states=np.array(xx),
                                    obstacle=[obs_x, obs_y, obs_diam/2.])

