"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/8/1 12:51
__file__ = P2C.py
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
import IntentionRecognition as ir
import airsim
import numpy as np
import random

ct = airsim.MultirotorClient()
ct.confirmConnection()

for i in range(80):
    sp = ir.SimulationPlatform(
        ct,
        np.array([
            [0, 0],
            [0, 15],
            [0, -15]
        ]),
        15,
        airsim.Vector3r(60, 0, -14)
    )

    sp.All_Armed_N_API()
    sp.All_Take_Off(-14)

    theta = random.randint(0, 3600) / 10.0
    r = random.randint(0, 4500) / 100.0
    sp.Cal_Goal(-14, theta, r)
    print(sp.goals)

    sp.Check_Intention()
    # sp.Plot_Routine()

    sp.Move_2_Goal_Together()
    sp.write()
    print('process: {} / 80'.format(i + 1))
    sp.ct.reset()