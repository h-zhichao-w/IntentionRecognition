"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/8/14 22:45
__file__ = DubinsRoutine.py
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

"""
This is a file intended to figure out a dubins routine with certain paraments,
also served as a support file for the project.
"""

import numpy as np

pi = np.pi
r = 1

def LSL(alpha, beta, d):
    tmp0 = d + np.sin(alpha) - np.sin(beta)
    p_squared = 2 + d**2 - (2 * np.cos(alpha - beta)) + (2 * d * (np.sin(alpha) - np.sin(beta)))
    if (p_squared < 0):
        L = np.array([np.inf, np.inf, np.inf, np.inf])
    else:
        tmp1 = np.arctan2((np.cos(beta) - np.cos(alpha)), tmp0)
        t = (-alpha + tmp1) % (2 * pi)
        p = p_squared**0.5
        q = (beta - tmp1) % (2 * pi)
        L = np.array([t + p + q, t, p, q])
    return L

def DubinsRoutine(p1: np.ndarray, p2: np.ndarray):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d = (dx**2 + dy**2 )**0.5 / r

    theta = (np.arctan2(dy, dx)) % (2 * pi)
    alpha = (p1[2] - theta) % (2 * pi)
    beta = (p2[2] - theta) % (2 * pi)

    L = np.zeros((6, 4))
    L[0] = LSL(alpha, beta, d)



