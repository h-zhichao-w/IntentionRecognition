"""
__title__ = ''
__author__ = 'Thompson'
__mtime__ = 2019/3/28
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃            ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import random
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def distance(A, B):
    a = A.x - B.x
    b = A.y - B.y
    return math.sqrt(a*a + b*b)

def square(A, B, C):
    a = distance(A, B)
    b = distance(A, C)
    c = distance(B, C)
    p = a + b + c
    return math.sqrt(p*(p-a)*(p-b)*(p-c))

def setax():
    ax = gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

d = []
g = []
for i in range(500):
    par = []
    for i in range(5):
        par.append(random.randint(-100, 100))
    print(par)

    # x = np.arange(0, 21)
    # y = par[0]*x*x + par[1]*x
    # z = par[2]*x*x - 30*par[2]*x
    # m = par[3]*x*x - 20*par[3]*x
    # plt.title("Matplotlib demo")
    # plt.xlabel("x axis caption")
    # plt.ylabel("y axis caption")
    # plt.plot(x, y, 'b')
    # plt.plot(x, z, 'g')
    # plt.plot(x, m, 'r')
    # setax()
    # plt.show()

    ly, lz, lm = [], [], []
    for i in range(21):
        ly.append(par[0]*i*i + par[1]*i)
        lz.append(par[2]*i*i - 30*par[2]*i)
        lm.append(par[3]*i*i - 20*par[3]*i)

    A1, B1, C1 = Point(0, ly[0]), Point(15, lz[15]), Point(20, lm[20])
    A2, B2, C2 = Point(4, ly[4]), Point(11, lz[11]), Point(16, lm[16])
    A3, B3, C3 = Point(8, ly[8]), Point(7, lz[7]), Point(12, lm[12])

    S1 = square(A1, B1, C1)
    S2 = square(A2, B2, C2)
    S3 = square(A3, B3, C3)
    d.append((S1-S2)/400000)
    g.append((S2-S3)/400000)

for i in range(len(d)):
    plt.scatter(d[i], g[i])
setax()
plt.title("Sample")
plt.xlabel("change in S within 3 seconds")
plt.ylabel("change in S within 5 seconds")
plt.show()