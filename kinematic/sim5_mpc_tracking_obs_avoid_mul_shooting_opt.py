import casadi as ca
import numpy as np
import math
import time

from draw import Draw_MPC_tracking_Obstacle

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
        if states[i+1, 2] > math.pi:
            states[i+1, 2] -= 2*math.pi
        elif states[i+1, 2] < math.pi:
            states[i+1, 2] += 2*math.pi
    return states

def desired_command_and_trajectory(t, T, x0_:np.array, N_):
    # initial state / last state
    x_ = np.zeros((N_+1, 3))
    x_[0] = x0_
    u_ = np.zeros((N_, 3))
    # states for the next N_ trajectories
    for i in range(N_):
        t_predict = t + T*i
        x_ref_ = 0.5 * t_predict
        y_ref_ = 1.0
        theta_ref_ = 0.0
        v_ref_ = 0.5
        omega_ref_ = 0.0
        if x_ref_ >= 12.0:
            x_ref_ = 12.0
            v_ref_ = 0.0
        x_[i+1] = np.array([x_ref_, y_ref_, theta_ref_])
        u_[i] = np.array([v_ref_, 0, omega_ref_])
    # return pose and command
    return x_, u_

def desired_command_and_trajectory2(t, T, x0_:np.array, N_):
    # initial state / last state
    x_ = np.zeros((N_+1, 3))
    x_[0] = x0_
    u_ = np.zeros((N_, 3))
    # states for the next N_ trajectories
    for i in range(N_):
        t_predict = t + T*i
        x_ref_ = 4*math.cos(2*math.pi/12*t_predict)
        y_ref_ = 4*math.sin(2*math.pi/12*t_predict)
        theta_ref_ = 2*math.pi/12*t_predict + math.pi/2
        
        dotx_ref_ = -2*math.pi/12*y_ref_
        doty_ref_ =  2*math.pi/12*x_ref_
        dotq_ref_ =  2*math.pi/12

        vx_ref_ = dotx_ref_*math.cos(dotq_ref_) + doty_ref_*math.sin(dotq_ref_)
        vy_ref_ = -dotx_ref_*math.sin(dotq_ref_) + doty_ref_*math.cos(dotq_ref_)
        omega_ref_ = dotq_ref_

        x_[i+1] = np.array([x_ref_, y_ref_, theta_ref_])
        u_[i] = np.array([vx_ref_, vy_ref_, omega_ref_])
    # return pose and command
    return x_, u_

if __name__ == "__main__":
    T = 0.1                 # time step
    N = 30                  # horizon length
    rob_diam = 0.3          # [m]
    v_max = 3.0             # linear velocity max
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

    # parameters, these parameters are the reference trajectories of the pose and inputs
    opt_u_ref = opti.parameter(N, 3)
    opt_x_ref = opti.parameter(N+1, 3)

    # create model
    f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2]) - u_[1]*ca.sin(x_[2]), u_[0]*ca.sin(x_[2]) + u_[1]*ca.cos(x_[2]), u_[2]])
    f_np = lambda x_, u_: np.array([u_[0]*ca.cos(x_[2]) - u_[1]*ca.sin(x_[2]), u_[0]*ca.sin(x_[2]) + u_[1]*ca.cos(x_[2]), u_[2]])

    # initial condition
    opti.subject_to(opt_states[0, :] == opt_x_ref[0, :])
    for i in range(N):
        x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T*T
        opti.subject_to(opt_states[i+1, :] == x_next)

    # obstacle
    obs_x = [2.0, 0.0, -4.0, 1.0]
    obs_y = [1.0, 4.0, 0.0, -4.0]
    obs_diam = 0.5
    bias = 0.05

    # add constraints to obstacle
    for i in range(N+1):
        for j in range(len(obs_x)):
            temp_constraints_ = ca.sqrt((opt_states[i, 0]-obs_x[j]-bias)**2+(opt_states[i, 1]-obs_y[j])**2-bias) - rob_diam - obs_diam/2
            opti.subject_to(opti.bounded(0.0, temp_constraints_, 10.0))

    # weight matrix
    Q = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 10.0]])
    R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]])

    # cost function
    obj = 0
    for i in range(N):
        state_error_ = opt_states[i, :] - opt_x_ref[i+1, :]
        control_error_ = opt_controls[i, :] - opt_u_ref[i, :]
        obj = obj + ca.mtimes([state_error_, Q, state_error_.T]) + ca.mtimes([control_error_, R, control_error_.T])
    opti.minimize(obj)

    #### boundrary and control conditions
    opti.subject_to(opti.bounded(-math.inf, x, math.inf))
    opti.subject_to(opti.bounded(-math.inf, y, math.inf))
    opti.subject_to(opti.bounded(-math.inf, theta, math.inf))
    opti.subject_to(opti.bounded(-v_max, vx, v_max))
    opti.subject_to(opti.bounded(-v_max, vy, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))

    opts_setting = {'ipopt.max_iter':2000,
                    'ipopt.print_level':0,
                    'print_time':0,
                    'ipopt.acceptable_tol':1e-8,
                    'ipopt.acceptable_obj_change_tol':1e-6}

    opti.solver('ipopt', opts_setting)

    t0 = 0
    init_state = np.array([0.0, 0.0, 0.0])
    current_state = init_state.copy()
    u0 = np.zeros((N, 3))
    next_trajectories = np.tile(init_state, N+1).reshape(N+1, -1) # set the initial state as the first trajectories for the robot
    next_controls = np.zeros((N, 3))
    next_states = np.zeros((N+1, 3))
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [t0] # for the time
    xx = []
    sim_time = 12.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    while(mpciter-sim_time/T<0.0):
        ## set parameter, here only update initial state of x (x0)
        opti.set_value(opt_x_ref, next_trajectories)
        opti.set_value(opt_u_ref, next_controls)
        ## provide the initial guess of the optimization targets
        opti.set_initial(opt_controls, u0.reshape(N, 3))# (N, 3)
        opti.set_initial(opt_states, next_states) # (N+1, 3)
        ## solve the problem once again
        t_ = time.time()
        sol = opti.solve()
        index_t.append(time.time()- t_)
        ## obtain the control input
        u_res = sol.value(opt_controls)
        x_m = sol.value(opt_states)
        # print(x_m[:3])
        u_c.append(u_res[0, :])
        t_c.append(t0)
        x_c.append(x_m)
        t0, current_state, u0, next_states = shift(T, t0, current_state, u_res, x_m, f_np)
        xx.append(current_state)
        ## estimate the new desired trajectories and controls
        next_trajectories, next_controls = desired_command_and_trajectory2(t0, T, current_state, N)
        mpciter = mpciter + 1


    ## after loop
    print(mpciter)
    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time)/(mpciter))
    ## draw function
    draw_result = Draw_MPC_tracking_Obstacle(rob_diam=0.3,
                                            init_state=init_state,
                                            robot_states=np.array(xx),
                                            obstacle=[obs_x, obs_y, obs_diam/2.],
                                            export_fig=False)