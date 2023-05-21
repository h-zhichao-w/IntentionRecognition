"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/8/20 19:16
__file__ = DubinsCal.py
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
import numpy as np
import random
import matplotlib.pyplot as plt
import NumericalAnalysis as na
import time

pi = np.pi

def Get_V():
    """
    :return: the velocity is assumed to be in the range of [0.95, 1.05] m/s1
    """
    return random.randint(195, 205) / 1000.00

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

def refactor(angle: float):
    if angle >= pi:
        angle -= (2.0 * pi)
    elif angle < -pi:
        angle += (2.0 * pi)
    return angle

def Get_Route(Kp: float, X: np.ndarray, Y: np.ndarray, Theta: np.ndarray, step: int, L = 1.5, delta_t = 0.1, drone_number = 3):
    """
    :param L: L is the length between the front and rear tires and is set to 1.5m
    :param Kp: the swarm's1 intention, 0.1 for contraction, -0.1 for extraction, and 0.0 for freestyle
    :param X: the x value of the start points
    :param Y: the y value of the start points
    :param Theta: the theta of the start points
    :param delta_t: sampling time
    :param step: number of iteration steps
    :return: the dubins route, listing out the coordinate of the points on the route
    """
    route = [
        np.array([[X[0]], [Y[0]], [D2R(Theta[0])], [Get_V()]]),    # the data (coordinates - 0,1, angle - 2, velocity - 3) of the drone 0
        np.array([[X[1]], [Y[1]], [D2R(Theta[1])], [Get_V()]]),    # the data (coordinates - 0,1, angle - 2, velocity - 3) of the drone 1
        np.array([[X[2]], [Y[2]], [D2R(Theta[2])], [Get_V()]])     # the data (coordinates - 0,1, angle - 2, velocity - 3) of the drone 2
    ]

    t = np.zeros(step + 1)  # 时间维度

    for i in range(step):
        t[i + 1] = delta_t * (i + 1)
        # 计算实时中心
        p_c = np.array([
            (route[0][0][-1] + route[1][0][-1] + route[2][0][-1]) / 3.0,
            (route[0][1][-1] + route[1][1][-1] + route[2][1][-1]) / 3.0
        ])

        for j in range(drone_number):
            obj = route[j]
            angle = refactor(obj[2][-1])

            v = obj[3][-1]
            theta_d = np.arctan2(p_c[1] - obj[1][-1], p_c[0] - obj[0][-1])
            alpha = refactor(theta_d - angle)

            u_phi = min(pi / 8.0, max(-pi / 8.0, Kp * alpha))
            x = obj[0][-1] + v * np.cos(obj[2][-1]) * delta_t + noise('cor')
            y = obj[1][-1] + v * np.sin(obj[2][-1]) * delta_t + noise('cor')
            theta = obj[2][-1] + v / L * np.tan(u_phi) * delta_t + noise('theta')

            route[j] = np.hstack((obj, [[x], [y], [theta], [Get_V()]]))
    route = np.array(route)

    return route, t

KP = [0.02, 0.01, 0.0, -0.01, -0.02]

# with open('model4.0/TRAIN-DATA-{}.csv'.format(time.strftime("%Y%m%d", time.localtime())), 'a') as f:
#     f.write('ax-0' + ',' + 'ay-0' + ',' + 'omega-0' + ',' + 'ax-1' + ',' + 'ay-1' + ',' + 'omega-1' + ',' + 'ax-2' + ',' + 'ay-2' + ',' + 'omega-2\n')
# with open('model4.0/TRAIN-LABEL-{}.csv'.format(time.strftime("%Y%m%d", time.localtime())), 'a') as f:
#     f.write('label\n')


for k in range(600):
    index = k % 3
    kp = KP[index]
    route, t = Get_Route(
        kp,   # 聚集：0.05，扩散：-0.02
        np.array([0., 4., 2.]),
        np.array([0., 0., 2*1.7]),
        np.array([0., 120., -120.]),
        20,
        L= 98e-3,
        delta_t= 0.8
    )

    for length in range(5, len(route[0, 0]) + 1, 5):
        for drone in range(3):
            # omega = np.zeros(length)
            v_x = np.zeros(length)
            v_y = np.zeros(length)
            # alpha = np.zeros(length - 1)
            a_x = np.zeros(length - 1)
            a_y = np.zeros(length - 1)

            for i in range(length):
                ipt = na.piecewiselinear_interploation(t, route[drone, 2], t[i] + 0.001)
                # omega[i] = (ipt - route[drone, 2, i]) / 0.001           # 角速度 - 方向角插值差商
                v_x[i] = route[drone, 3, i] * np.cos(route[drone, 2, i])    # x方向速度
                v_y[i] = route[drone, 3, i] * np.sin(route[drone, 2, i])    # y方向速度

            for i in range(length - 1):
                # ipt_om = na.piecewiselinear_interploation(t[ : half_len], omega, t[i] + 0.001)
                ipt_vx = na.piecewiselinear_interploation(t[: length], v_x, t[i] + 0.001)
                ipt_vy = na.piecewiselinear_interploation(t[ : length], v_y, t[i] + 0.001)
                # alpha[i] = (ipt_om - omega[i]) / 0.001
                a_x[i] = (ipt_vx -v_x[i]) / 0.001   # x方向加速度 - x方向速度插值差商
                a_y[i] = (ipt_vy - v_y[i]) / 0.001  # y方向加速度 - y方向速度插值差商

            print(np.average(a_x), np.average(a_y))

            with open('model4.0/TRAIN-DATA-{}.csv'.format(time.strftime("%Y%m%d", time.localtime())), 'a') as f:
                f.write(str(np.average(a_x)) + ',' + str(np.average(a_y)))
                if drone != 2:
                    f.write(',')
                else:
                    f.write('\n')

        with open('model4.0/TRAIN-LABEL-{}.csv'.format(time.strftime("%Y%m%d", time.localtime())), 'a') as f:
            f.write('1\n')
