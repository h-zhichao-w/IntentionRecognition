"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/8/9 19:05
__file__ = test.py
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
import random
import numpy as np

ct = airsim.MultirotorClient()
ct.confirmConnection()


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

sp.Move_2_Goal_Together()

sp.Check_Intention()
sp.Plot_Routine()

sp.ct.reset()
