"""
__title__ = 'test.py'
__author__ = 'Hansen Wong // 王之超'
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
import random
import math

velocity = 10
def get_v():
    noise = random.randint(-30, 30) * 0.1
    return velocity + noise

def GeneralEquation(first_x, first_y, second_x, second_y):
    # 一般式 Ax+By+C=0
    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    return A, B, C

def Distance(first_x, first_y, second_x, second_y):
    d = math.sqrt((first_x - second_x)**2 + (first_y - second_y)**2)
    return d

def Mid_point(first_x, first_y, second_x, second_y):
    mid_x = (first_x + second_x) / 2
    mid_y = (first_y + second_y) / 2
    return mid_x, mid_y

class Drone:
    x, y = 0, 0
    tx, ty = 0, 0
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Drones:
    def __init__(self, ini):
        """
        输入无人机编组的初始阵型，通过随机数函数生成飞机的坐标
        pattern: h-horizontal, v-vertical, t-triangle, l-left, r-right
        """
        self.ini = ini
        self.A = Drone(0, 0)
        if self.ini == 'h':
            rand = random.randint(50, 101)
            self.B = Drone(rand, 0)
            self.C = Drone(2 * rand, 0)
        print("Drone A created, initial cor ({}, {})\n"
              "Drone B created, initial cor ({}, {})\n"
              "Drone C created, initial cor ({}, {})\n"
              .format(self.A.x, self.A.y, self.B.x, self.B.y, self.C.x, self.C.y))

    def set_target(self, tar):
        if tar == self.ini:
            print("The initial and target pattern are the same!")
            return

        if tar == 'v':
            rand = random.randint(100, 201)
            self.B.tx, self.B.ty = self.B.x, -rand
            self.A.tx, self.A.ty = self.B.x, -2 * rand
            self.C.tx, self.C.ty = self.B.x, -3 * rand
            self.tag = [0]

        elif tar == 't':
            rand = random.randint(100, 201)
            self.B.tx, self.B.ty = self.B.x, -rand
            self.A.tx, self.A.ty = self.B.tx - (rand - 100)/1.732, -100
            self.C.tx, self.C.ty = self.B.tx + (rand - 100)/1.732, -100
            self.tag = [1]

        elif tar == 'r':
            rand1 = random.randint(100, 201)
            rand2 = random.randint(100, 201)

            self.A.tx = random.randint(-100, self.B.x + 1)
            self.A.ty = -random.randint(100, 201)
            self.B.tx = self.A.tx + rand1
            self.B.ty = self.A.ty - rand2
            self.C.tx = self.B.tx + rand1
            self.C.ty = self.B.ty - rand2
            self.tag = [2]

        elif tar == 'l':
            rand1 = random.randint(100, 201)
            rand2 = random.randint(100, 201)

            self.C.tx = random.randint(self.B.x, self.C.x + 101)
            self.C.ty = -random.randint(100, 201)
            self.B.tx = self.C.tx - rand1
            self.B.ty = self.C.ty - rand2
            self.A.tx = self.B.tx - rand1
            self.A.ty = self.B.ty - rand2
            self.tag = [3]

        else:
            print("阵型输入错误")

        self.A.d = Distance(self.A.x, self.A.y, self.A.tx, self.A.ty)
        self.B.d = Distance(self.B.x, self.B.y, self.B.tx, self.B.ty)
        self.C.d = Distance(self.C.x, self.C.y, self.C.tx, self.C.ty)
        self.d_max = max(self.A.d, self.B.d, self.C.d)
        self.T = self.d_max / get_v()

        print("Drone A's target ({}, {})\n"
              "Drone B's target ({}, {})\n"
              "Drone C's target ({}, {})\n"
              .format(self.A.tx, self.A.ty, self.B.tx, self.B.ty, self.C.tx, self.C.ty))

    def show(self):
        """
        展示运动轨迹
        """
        plt.plot([self.A.x, self.A.tx], [self.A.y, self.A.ty], 'r')
        plt.plot([self.B.x, self.B.tx], [self.B.y, self.B.ty], 'g')
        plt.plot([self.C.x, self.C.tx], [self.C.y, self.C.ty], 'b')
        plt.show()

    def calulate(self):
        """
        计算三个无人机的运动方程，通过运动方程计算每个时刻的速度和坐标
        """
        EquationA = {}
        EquationA['A'], EquationA['B'], EquationA['C'] = GeneralEquation(self.A.x, self.A.y, self.A.tx, self.A.ty)

        EquationB = {}
        EquationB['A'], EquationB['B'], EquationB['C'] = GeneralEquation(self.B.x, self.B.y, self.B.tx, self.B.ty)

        EquationC = {}
        EquationC['A'], EquationC['B'], EquationC['C'] = GeneralEquation(self.C.x, self.C.y, self.C.tx, self.C.ty)

        print("A的运动方程为{}x+{}y+{}=0\n"
              "B的运动方程为{}x+{}y+{}=0\n"
              "C的运动方程为{}x+{}y+{}=0\n".format(
            EquationA['A'], EquationA['B'], EquationA['C'],
            EquationB['A'], EquationB['B'], EquationB['C'],
            EquationC['A'], EquationC['B'], EquationC['C'])
        )

        t = 0
        Vax, Vay, Vbx, Vby, Vcx, Vcy = [], [], [], [], [], []
        Dab, Dbc, Dac = [], [], []
        Mabx, Mbcx, Macx = [], [], []
        Maby, Mbcy, Macy = [], [], []
        while (t < self.T/3):
            self.A.vx = -get_v() * EquationA['B'] / math.sqrt(EquationA['A'] * EquationA['A'] + EquationA['B'] * EquationA['B'])
            self.A.vy = get_v() * EquationA['A'] / math.sqrt(EquationA['A'] * EquationA['A'] + EquationA['B'] * EquationA['B'])

            self.B.vx = -get_v() * EquationB['B'] / math.sqrt(EquationB['A'] * EquationB['A'] + EquationB['B'] * EquationB['B'])
            self.B.vy = get_v() * EquationB['A'] / math.sqrt(EquationB['A'] * EquationB['A'] + EquationB['B'] * EquationB['B'])

            self.C.vx = -get_v() * EquationC['B'] / math.sqrt(EquationC['A'] * EquationC['A'] + EquationC['B'] * EquationC['B'])
            self.C.vy = get_v() * EquationC['A'] / math.sqrt(EquationC['A'] * EquationC['A'] + EquationC['B'] * EquationC['B'])

            if (self.A.y <= self.A.ty): self.A.vx, self.A.vy = 0, 0
            if (self.B.y <= self.B.ty): self.B.vx, self.B.vy = 0, 0
            if (self.C.y <= self.C.ty): self.C.vx, self.C.vy = 0, 0

            Vax.append(self.A.vx)
            Vay.append(self.A.vy)
            Vbx.append(self.B.vx)
            Vby.append(self.B.vy)
            Vcx.append(self.C.vx)
            Vcy.append(self.C.vy)

            self.A.x += (0.1 * self.A.vx)
            self.A.y += (0.1 * self.A.vy)
            self.B.x += (0.1 * self.B.vx)
            self.B.y += (0.1 * self.B.vy)
            self.C.x += (0.1 * self.C.vx)
            self.C.y += (0.1 * self.C.vy)

            Dab.append(Distance(self.A.x, self.A.y, self.B.x, self.B.y))
            Dbc.append(Distance(self.B.x, self.B.y, self.C.x, self.C.y))
            Dac.append(Distance(self.A.x, self.A.y, self.C.x, self.C.y))

            Mabx.append(Mid_point(self.A.x, self.A.y, self.B.x, self.B.y)[0])
            Mbcx.append(Mid_point(self.B.x, self.B.y, self.C.x, self.C.y)[0])
            Macx.append(Mid_point(self.A.x, self.A.y, self.C.x, self.C.y)[0])

            Maby.append(Mid_point(self.A.x, self.A.y, self.B.x, self.B.y)[1])
            Mbcy.append(Mid_point(self.B.x, self.B.y, self.C.x, self.C.y)[1])
            Macy.append(Mid_point(self.A.x, self.A.y, self.C.x, self.C.y)[1])

            t += 0.1

        data = np.array([Dab, Dbc, Dac, Vax, Vbx, Vcx, Vay, Vby, Vcy, Mabx, Mbcx, Macx, Maby, Mbcy, Macy])
        average = np.mean(data, axis=1)
        print(average)

        with open('data.csv', 'a') as f:
            for i in average:
                f.write(str(i) + ',')
            f.write(str(self.tag[0]))
            f.write('\n')

i = 1
intru = ['v', 't', 'r', 'l']
count = {'v':0, 't':0, 'r':0, 'l':0}

while (i <= 20000):

    group = Drones('h')
    goal = random.randint(0, 3)
    group.set_target(intru[goal])
    group.calulate()

    count[intru[goal]] += 1
    i += 1

print("共记录h->v{}条数据，h->t{}条数据，h->r{}条数据，h->l{}条数据".format(
    count['v'], count['t'], count['r'], count['l']
))