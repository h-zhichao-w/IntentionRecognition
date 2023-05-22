"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/8/21 10:17
__file__ = dubins_sim.py
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
import dubins2 as db
import matplotlib.pyplot as plt
import airsim
import IntentionRecognition as ir

for a in range(81):
    angle = a / 2.
    print(angle)
    for it in range(-1, 2):
        kp = it / 10.
        route = db.Get_Route(
            kp,
            np.array([0., 12., 6.]),
            np.array([0., 0., 10.39]),
            np.array([40. + angle, 160. + angle, -80. + angle]),
            # theta's ranges in the dubins2.py are 0 ~ 60, 120 ~ 180, -120 ~ -60.
            100
        )

        print(route)

        for i in range(3):
            for j in route[i]:
                plt.plot(j[0], j[1], 'ob')
        plt.title('Kp = {}, angle(drone0) = {}'.format(kp, 40. + angle))
        plt.draw()
        plt.pause(0.5)
        plt.close(1)

        height = -3
        R = [[], [], []]
        for i in range(3):
            for j in route[i]:
                R[i].append(airsim.Vector3r(j[1], j[0], height))

        ct = airsim.MultirotorClient()
        ct.confirmConnection()

        sp = ir.SimulationPlatform(
            ct,
            np.array([
                [0., 0.],
                [0., 12.],
                [10.39, 6.]
            ]),
            1.,
            kp
        )

        sp.All_Armed_N_API()
        sp.All_Take_Off(height)

        sp.Move_On_Path_Together(R)

        sp.write('data_du.csv', 'label_du.csv')

        sp.ct.reset()