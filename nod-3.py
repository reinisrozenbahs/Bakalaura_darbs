import torch
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
plt.ion()

target_point = np.array([-3.0, 0])
anchor_point = np.array([0, 0])

is_running = True
def button_press_event(event):
    global target_point
    target_point = np.array([event.xdata, event.ydata])

def press(event):
    global is_running
    print('press', event.key)
    if event.key == 'escape':
        is_running = False # quits app

fig, _ = plt.subplots()
fig.canvas.mpl_connect('button_press_event', button_press_event)
fig.canvas.mpl_connect('key_press_event', press)

length_joint = 2.0
theta_1 = np.deg2rad(-10)
theta_2 = np.deg2rad(-10)
step = 0.01

def rotation(theta):
    c=np.cos(theta)
    s=np.sin(theta)
    R = np.array([
        [c, -s],
        [s, c]
    ])
    return R

def d_rotation(theta):
    c=np.cos(theta)
    s=np.sin(theta)
    dR = np.array([
        [-s, -c],
        [c, -s]
    ])
    return dR

while is_running:
    plt.clf()
    plt.title(f'theta_1: {round(np.rad2deg(theta_1))} theta_2: {round(np.rad2deg(theta_2))}')

    R1 = rotation(theta_1)
    R2 = rotation(theta_2)
    dR1 = d_rotation(theta_1)
    dR2 = d_rotation(theta_2)

    segment = np.array([0.0, 1.0])*length_joint

    joints = []
    joints.append(anchor_point)
    joint1 = R1 @ segment
    joint2 = R1 @ (segment + R2 @ segment)
    joints.append(joint1)
    joints.append(joint2)

    #L_mse = np.sum((target_point-joint2)**2)

    d_theta1 = np.sum(dR1 @ segment * -2 * (target_point - joint2))
    d_theta2 = np.sum(R1 @ dR2 @ segment * -2 * (target_point - joint2))

    theta_1 -= d_theta1 * step
    theta_2 -= d_theta2 * step

    np_joints = np.array(joints)

    if len(np_joints):
        plt.plot(np_joints[:, 0], np_joints[:, 1])
    plt.scatter(target_point[0], target_point[1], s=50, c='r')

    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    plt.draw()
    plt.pause(1e-3)
    #break
input('end')