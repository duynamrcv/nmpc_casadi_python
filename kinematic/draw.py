#!/usr/bin/env python
# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches


class Draw_MPC_point_stabilization_v1(object):
    def __init__(self, robot_states: np.array, init_state: np.array, target_state: np.array, rob_diam=0.3,
                 export_fig=False):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-0.8, 5), ylim=(-0.8, 5.))
        # self.fig.set_dpi(400)
        self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        if export_fig:
            self.ani.save('./v1.gif', writer='imagemagick', fps=100)
        plt.show()

    def animation_init(self):
        # plot target state
        self.target_circle = plt.Circle(self.target_state[:2], self.rob_radius, color='b', fill=False)
        self.ax.add_artist(self.target_circle)
        self.target_arr = mpatches.Arrow(self.target_state[0], self.target_state[1],
                                         self.rob_radius * np.cos(self.target_state[2]),
                                         self.rob_radius * np.sin(self.target_state[2]), width=0.2)
        self.ax.add_patch(self.target_arr)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.target_circle, self.target_arr, self.robot_body, self.robot_arr

    def animation_loop(self, indx):
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]
        self.robot_body.center = position
        # self.ax.add_artist(self.robot_body)
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        self.ax.plot(self.robot_states[:indx, 0], self.robot_states[:indx, 1], color='r', linewidth=1.5)
        return self.robot_arr, self.robot_body


class Draw_MPC_Obstacle(object):
    def __init__(self, robot_states: np.array, init_state: np.array, target_state: np.array, obstacle: list,
                 rob_diam=0.3, export_fig=False):
        self.robot_states = robot_states
        self.init_state = init_state
        self.target_state = target_state
        self.rob_radius = rob_diam / 2.0
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-0.8, 5), ylim=(-0.8, 5.))
        if obstacle is not None:
            self.obstacle = obstacle
        else:
            print('no obstacle given, break')
        self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        if export_fig:
            self.ani.save('obstacle.gif', writer='imagemagick', fps=100)
        plt.show()

    def animation_init(self):
        # plot target state
        self.target_circle = plt.Circle(self.target_state[:2], self.rob_radius, color='b', fill=False)
        self.ax.add_artist(self.target_circle)
        self.target_arr = mpatches.Arrow(self.target_state[0], self.target_state[1],
                                         self.rob_radius * np.cos(self.target_state[2]),
                                         self.rob_radius * np.sin(self.target_state[2]), width=0.2)
        self.ax.add_patch(self.target_arr)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        self.obstacle_circle = []
        for i in range(len(self.obstacle[0])):
            center = (self.obstacle[0][i], self.obstacle[1][i])
            obs = plt.Circle(center, self.obstacle[2], color='g', fill=True)
            self.obstacle_circle.append(obs)
            self.ax.add_artist(obs)
        
        return self.target_circle, self.target_arr, self.robot_body, self.robot_arr, self.obstacle_circle

    def animation_loop(self, indx):
        position = self.robot_states[indx][:2]
        orientation = self.robot_states[indx][2]
        self.robot_body.center = position
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.plot(self.robot_states[:indx, 0], self.robot_states[:indx, 1], color='r', linewidth=1.5)

        self.ax.add_patch(self.robot_arr)
        return self.robot_arr, self.robot_body


class Draw_MPC_tracking(object):
    def __init__(self, robot_states: np.array, init_state: np.array, rob_diam=0.3, export_fig=False):
        self.init_state = init_state
        self.robot_states = robot_states
        self.rob_radius = rob_diam
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-6.0, 6.0), ylim=(-6.0, 6.0))
        # self.fig.set_size_inches(7, 6.5)
        # init for plot
        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        if export_fig:
            self.ani.save('tracking.gif', writer='imagemagick', fps=100)
        plt.show()

    def animation_init(self, ):
        # draw target line
        step =  np.arange(0,12.1,np.pi/50)
        x_ref = 4*np.cos(step)
        y_ref = 4*np.sin(step)
        self.target_line = plt.plot(x_ref, y_ref, '-b')
        # draw the initial position of the robot
        self.init_robot_position = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.init_robot_position)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        return self.target_line, self.init_robot_position, self.robot_body, self.robot_arr

    def animation_loop(self, indx):
        position = self.robot_states[indx,:2]
        orientation = self.robot_states[indx,2]
        self.robot_body.center = position
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        self.ax.plot(self.robot_states[:indx, 0], self.robot_states[:indx, 1], color='r', linewidth=1.5)
        return self.robot_arr, self.robot_body

class Draw_MPC_tracking_Obstacle(object):
    def __init__(self, robot_states: np.array, init_state: np.array, obstacle: list, rob_diam=0.3, export_fig=False):
        self.init_state = init_state
        self.robot_states = robot_states
        self.rob_radius = rob_diam
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-6.0, 6.0), ylim=(-6.0, 6.0))
        if obstacle is not None:
            self.obstacle = obstacle
        else:
            print('no obstacle given, break')

        self.animation_init()

        self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
                                           init_func=self.animation_init, interval=100, repeat=False)

        plt.grid('--')
        if export_fig:
            self.ani.save('tracking_obs_avoid.gif', writer='imagemagick', fps=100)
        plt.show()

    def animation_init(self, ):
        # draw target line
        step =  np.arange(0,12.1,np.pi/50)
        x_ref = 4*np.cos(step)
        y_ref = 4*np.sin(step)
        self.target_line = plt.plot(x_ref, y_ref, '-b')
        # draw the initial position of the robot
        self.init_robot_position = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.init_robot_position)
        self.robot_body = plt.Circle(self.init_state[:2], self.rob_radius, color='r', fill=False)
        self.ax.add_artist(self.robot_body)
        self.robot_arr = mpatches.Arrow(self.init_state[0], self.init_state[1],
                                        self.rob_radius * np.cos(self.init_state[2]),
                                        self.rob_radius * np.sin(self.init_state[2]), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        self.obstacle_circle = []
        for i in range(len(self.obstacle[0])):
            center = (self.obstacle[0][i], self.obstacle[1][i])
            obs = plt.Circle(center, self.obstacle[2], color='g', fill=True)
            self.obstacle_circle.append(obs)
            self.ax.add_artist(obs)
        return self.target_line, self.init_robot_position, self.robot_body, self.robot_arr, self.obstacle_circle

    def animation_loop(self, indx):
        position = self.robot_states[indx,:2]
        orientation = self.robot_states[indx,2]
        self.robot_body.center = position
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(position[0], position[1], self.rob_radius * np.cos(orientation),
                                        self.rob_radius * np.sin(orientation), width=0.2, color='r')
        self.ax.add_patch(self.robot_arr)
        self.ax.plot(self.robot_states[:indx, 0], self.robot_states[:indx, 1], color='r', linewidth=1.5)
        return self.robot_arr, self.robot_body