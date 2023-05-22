"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/8/20 19:16
__file__ = dubins2.py
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓    ┏┓
            ┏┛┻━━━┛ ┻┓
            ┃         ┃
            ┃  ┳┛  ┗┳  ┃
            ┃     ┻    ┃
            ┗━┓      ┏━┛
              ┃      ┗━━━┓
              ┃  神兽保佑  ┣┓
              ┃　永无BUG！ ┏┛
                ┗┓┓┏━┳┓┏┛
                 ┃┫┫  ┃┫┫
                 ┗┻┛  ┗┻┛
"""
import matplotlib.pyplot as plt
import numpy as np
import random

pi = np.pi

def Get_V():
    """
    :return: the velocity is assumed to be in the range of [0.95, 1.05] m/s
    """
    return random.randint(950, 1050) / 1000.00

def D2R(degree: float):
    r = degree * pi / 180.0
    return r

def R2D(r: float):
    d = r * 180.0 / pi
    return d

def noise(type: str):
    if type == 'cor':
        return random.randint(-100, 100) / 10000.
    else:
        return random.randint(-67, 67) / 10000.

def Get_Route(Kp: float, X: np.ndarray, Y: np.ndarray, Theta: np.ndarray, step: int, L = 1.5, delta_t = 0.1, drone_number = 3):
    """
    :param L: L is the length between the front and rear tires and is set to 1.5m
    :param Kp: the swarm's intention, 0.1 for contraction, -0.1 for extraction, and 0.0 for freestyle
    :param X: the x value of the start points
    :param Y: the y value of the start points
    :param Theta: the theta of the start points
    :param delta_t: sampling time
    :param step: number of iteration steps
    :return: the dubins route, listing out the coordinate of the points on the route
    """
    route = [
        np.array([[X[0], Y[0], Get_V()]]),
        np.array([[X[1], Y[1], Get_V()]]),
        np.array([[X[2], Y[2], Get_V()]])
    ]

    for i in range(drone_number):
        Theta[i] = D2R(Theta[i])

    for i in range(step):
        for j in range(drone_number):
            if Theta[j] >= pi:
                # np.arctan2 returns a real number in the range of [-pi, pi]. as a result, we turn the theta into the same range
                Theta[j] -= (2.0 * pi)
            elif Theta[j] < -pi:
                Theta[j] += (2.0 * pi)

        p_c = np.array([
            (route[0][-1][0] + route[1][-1][0] + route[2][-1][0]) / 3.0,
            (route[0][-1][1] + route[1][-1][1] + route[2][-1][1]) / 3.0
        ])

        for j in range(drone_number):
            v = Get_V()
            theta_d = np.arctan2(p_c[1] - route[j][-1][1], p_c[0] - route[j][-1][0])
            alpha = theta_d - Theta[j]
            if alpha >= pi:
                alpha -= 2 * pi
            elif alpha < -pi:
                alpha += 2 * pi
            u_phi = min(pi / 8.0, max(-pi / 8.0, Kp * alpha))
            x = route[j][-1][0] + v * np.cos(Theta[j]) * delta_t + noise('cor')
            y = route[j][-1][1] + v * np.sin(Theta[j]) * delta_t + noise('cor')
            Theta[j] = Theta[j] + v / L * np.tan(u_phi) * delta_t + noise('theta')
            route[j] = np.vstack((route[j], [x, y, v]))
    route = np.array(route)
    return route