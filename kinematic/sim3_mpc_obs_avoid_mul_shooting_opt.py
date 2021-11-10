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
    # define prediction horizon function
    states = np.zeros((N+1, 3))
    states[0,:] = x0
    # euler method
    for i in range(N):
        states[i+1, 0] = states[i, 0] + (u[i, 0]*np.cos(states[i, 2]) - u[i, 1]*np.sin(states[i, 2]))*T
        states[i+1, 1] = states[i, 1] + (u[i, 0]*np.sin(states[i, 2]) + u[i, 1]*np.cos(states[i, 2]))*T
        states[i+1, 2] = states[i, 2] + u[i, 2]*T
    return states

if __name__ == "__main__":
    T = 0.2                 # time step
    N = 30                  # horizon length
    rob_diam = 0.3          # [m]
    v_max = 1.0             # linear velocity max
    omega_max = np.pi/3.0   # angular velocity max

    opti = ca.Opti()
    # control variables, liear velocity and angular velocity
    opt_controls = opti.variable(N, 3)
    vx = opt_controls[:, 0]
    vy = opt_controls[:, 1]
    omega = opt_controls[:, 2]
    
    opt_states = opti.variable(N+1, 3)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]

    # parameters
    opt_x0 = opti.parameter(3)
    opt_xs = opti.parameter(3)

    # create model
    f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]) - u_[1]*ca.sin(x_[2]), u_[0]*ca.sin(x_[2]) + u_[1]*ca.cos(x_[2]), u_[2]])
    f_np = lambda x_, u_: np.array([u_[0]*ca.cos(x_[2]) - u_[1]*ca.sin(x_[2]), u_[0]*ca.sin(x_[2]) + u_[1]*ca.cos(x_[2]), u_[2]])

    # initial condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :] == x_next)

    # obstacle
    obs_x = [0.5, 1.5, 3.0, 3.0, 3.5, 4.5]
    obs_y = [1.0, 1.0, 1.5, 2.7, 4.0, 4.0]
    obs_diam = 0.5
    bias = 0.02

    # add constraints to obstacle
    for i in range(N+1):
        for j in range(len(obs_x)):
            temp_constraints_ = ca.sqrt((opt_states[i, 0]-obs_x[j]-bias)**2+(opt_states[i, 1]-obs_y[j])**2-bias)-rob_diam/2.0-obs_diam/2.0
            opti.subject_to(opti.bounded(0.0, temp_constraints_, 10.0))
        

    # weight matrix
    Q = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]])
    R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]])

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

    opts_setting = {'ipopt.max_iter': 100,
                    'ipopt.print_level': 0,
                    'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}
    opti.solver('ipopt', opts_setting)

    # The final state
    final_state = np.array([4.5, 4.5, 0.0])
    opti.set_value(opt_xs, final_state)

    # The initial state
    t0 = 0
    init_state = np.array([0.0, 0.0, 0.0])
    u0 = np.zeros((N, 3))
    current_state = init_state.copy()
    next_states = np.zeros((N+1, 3))
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
        opti.set_initial(opt_states, next_states) # (N+1, 3)
        
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

