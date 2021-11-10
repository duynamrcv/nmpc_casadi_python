# Nonlinear MPC using CasADi and Python

Applying Nonlinear MPC for Mobile Robot in simulation using Python3

## Robot model
Model: Three Omni-wheeled directional Mobile robot<br>
State: $ [x, y, \theta]^T $<br>
Control: $ [v_x, v_y, \omega]^T $<br>
Constraints:
$$ -1.0 \leq v_x, v_y \leq 1.0 $$
$$ -\frac{\pi}{3} \leq \omega \leq \frac{\pi}{3} $$

## Tasks:
* Sim 2 - Moving to target position
* Sim 3 - Moving to target position and avoid static obstacles
* Sim 4 - Tracking the desired path
* Sim 5 - Tracking the desired path and avoid static obstacles

## Results
### Sim 2
![sim2](results/sim2.png)
### Sim 3
![sim3](results/sim3.png)
![sim3](results/obstacle.gif)
### Sim 4
![sim4](results/sim4.png)
### Sim 5
![sim5](results/sim5.png)
![sim5](results/tracking_obs_avoid.gif)