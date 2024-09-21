
import matplotlib.pyplot as plt
import numpy as np
from RoverState import RoverState
import math
import SoftSensors
from tqdm import tqdm

import ExtendedKalmanFilter
import EKF


# plt.plot(np.linspace(0,1, N), np.random.normal(160, 170, size=N));
state = RoverState()
mag = SoftSensors.SoftCompass(0.005, state)
accelerometer = SoftSensors.SoftAccelerometer(0.0003, state)
gps = SoftSensors.SoftGPS(5, state)
gyro = SoftSensors.SoftGyro(1, state)
speed = SoftSensors.SoftSpeedSensor(2, state)

def normalize_angle(s):
    s %= math.pi * 2
    if(s > math.pi):
        s -= math.pi * 2
    return s

t_points = []
x_points = []
y_points = []
velocities = []
headings = []
heading_rates = []
ac_points = []
at_points = []


velocities_measured = []
x_points_measured = []
y_points_measured = []
headings_measured = []
heading_rates_measured = []
at_measured = []
ac_measured = []

x_ekf = []
x_var = []

headings_ekf = []
headings_var = []
 
v_ekf = []
a_ekf = []

t = 0
last_update = 0
sensor_dt = 1 / 24# 1/24
sim_dt = 0.001
# state.set_control(1, 1)
T = 500


state_history = []



P = np.diag([ 1000.0, 1000.0, 100.0, 100.0, 100.0 ])
# N = np.diag([ 0.01**2, 0.01**2, 0.01**2, 0.01**2, 0.01**2 ])
R = np.diag([
    gps.s**2, gps.s**2, mag.s**2, accelerometer.s**2
])

ekf = ExtendedKalmanFilter.AdaptiveRoverEKF(sensor_dt, P, R)
# ekf.X = np.array([[0, 0, 0]]).T
# ekf.R = np.diag([5**2,5**2, 0.005**2])
# ekf.P = np.diag([100,100,1000])


command_history = []

ekf_history = []
ekf_var_history = []

commands = [
    (100, 10000000, 0),
    # # (20, 10000, 0.5),
    # (50, 10, 0.2),
    # # (40, 10000, 0.5),
    # (50, 100, 0),
    # (75, 100, 0.2),
    # (100, 10000000, 0.2),
    # (200, -100, 0.5),
    (200, 1, 0.1),
    (300, -1, 0.1),
    # (100, 10000000, 2),
    (1000, 10000000, 0),
]


model_hist = []

# state.set_control(10, 1)
# ekf.set_control(1, 1)
# print(ekf.P)
curr_command = 0
for t in tqdm(np.arange(0, T, sim_dt)):

    t += sim_dt
    command_t, command_R, command_V = commands[curr_command]
    if(command_t < t):
        curr_command += 1
        command_t, command_R, command_V = commands[curr_command]


    state.set_control(command_R, command_V)
    state.update(sim_dt)
    if(sensor_dt < (t - last_update)):
        last_update = t
        t_points.append(t)
        state_history.append(state.x)
        x_points.append(state.position[0])
        y_points.append(state.position[1])
        headings.append(normalize_angle(state.heading))
        at_points.append(state.acceleration[0])
        ac_points.append(state.acceleration[1])
        velocities.append(state.velocity)
        
        a_read = accelerometer.read()
        gps_read = gps.read()
        a = mag.read()
        headings_measured.append(a)
        x_points_measured.append(gps_read[0])
        y_points_measured.append(gps_read[1])
        # heading_rates_measured.append(gyro.read())
        at_measured.append(a_read)
        # ac_measured.append(a_read[1])
        velocities_measured.append(speed.read())

        command = np.array([command_V, command_R])
        command_history.append(command)
        ekf.predict(command)
        # print(ekf.x)
        ekf.update(np.array([gps_read[0], gps_read[1], a, a_read[0]]))

        x_ekf.append(ekf.x[0])
        headings_ekf.append(normalize_angle(ekf.x[2]))
        v_ekf.append(ekf.x[3])
        # print(ekf.x[2,0])
        # print(ekf.P)
        headings_var.append(ekf.P[2, 2])
        x_var.append(ekf.P[0, 0])

        ekf_history.append(ekf.x)
        ekf_var_history.append(ekf.P)
        model_hist.append(ekf.model_i)

        # print()
        # print(ekf.x)


fig, axs = plt.subplots(4, 4, figsize=(12, 10))


def get_state_variable_history(i:int):
    return [ x[i] for x in state_history ]

def get_var(a:list, i:int):
    return [ x[i] for x in a ]

# plt.plot(x_points, y_points);
# plt.scatter(x_points_measured, y_points_measured);
# plt.scatter(t_points, headings_measured);
# plt.plot(t_points, headings);
# print(headings_ekf)
# print(headings)
# print(state_history)
axs[0,0].plot(t_points, get_state_variable_history(0))
axs[0,0].scatter(t_points, get_var(ekf_history, 0))
axs[0,0].scatter(t_points, x_points_measured)
axs[1,0].scatter(t_points, [a - b for a,b in zip(get_state_variable_history(0), get_var(ekf_history, 0))])
axs[2,0].scatter(t_points, x_var)
# axs[0,1].scatter(t_points, x);
axs[0,1].plot(t_points, get_state_variable_history(3))
axs[0,1].scatter(t_points, get_var(at_measured, 0))
axs[0,1].plot(t_points, get_state_variable_history(4))
axs[0,1].scatter(t_points, get_var(ekf_history, 3))
axs[0,1].scatter(t_points, get_var(ekf_history, 4))
axs[0,1].plot(t_points, get_var(command_history, 0))
# axs[1,1].plot(t_points, get_var(ekf_history, 3))
axs[2,1].plot(t_points, get_var(ekf_var_history, (3, 3)))
axs[1,1].plot(t_points, model_hist)
# axs[0,1].plot(t_points, y_points)

# axs[0,2].plot(t_points, [ 1/x for x in get_state_variable_history(5)])
axs[0,2].plot(t_points, get_state_variable_history(2), label="True Head")
a = axs[0,2].scatter(t_points, headings_ekf, label="Head EKF")
b = axs[0,2].scatter(t_points, headings_measured, label="Mag Read")
axs[0,2].legend(handles=[a, b])
# axs[0,2].plot(t_points, headings)
# axs[0,2].scatter(t_points, headings_ekf);
axs[1,2].scatter(t_points, [normalize_angle(a - b) for a,b in zip(get_state_variable_history(2), headings_ekf)])
axs[2,2].scatter(t_points, headings_var)

def dist(x1, y1, x2, y2):
    return [ math.sqrt((a -c)**2 + (b-d) ** 2) for a,b,c,d in zip(x1, y1, x2, y2)]


axs[3,2].scatter(t_points, dist(get_var(ekf_history,0),get_var(ekf_history,1), get_state_variable_history(0),get_state_variable_history(1)))
axs[3,3].plot(get_state_variable_history(0),get_state_variable_history(1))
axs[3,3].scatter(get_var(ekf_history,0),get_var(ekf_history,1))
# print(headings_ekf)
# plt.plot(t_points, headings);

# plt.scatter(t_points, velocities_measured);
# plt.plot(t_points, velocities);

# plt.scatter(t_points, at_measured);
# plt.plot(t_points, at_points);

plt.show()
# plt.wai