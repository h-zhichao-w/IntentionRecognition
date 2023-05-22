"""
__title__ = ''
__author__ = 'Thompson'
__mtime__ = 2019/3/28
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
test python environment
"""

import airsim
import time

# connect to the AirSim simulator
client = airsim.MultirotorClient()
# check the connection
client.confirmConnection()

# get control
client.enableApiControl(True)

# unlock
client.armDisarm(True)

"""
很多无人机或者汽车控制的函数都有 Async 作为后缀，这些函数在执行的时候会立即返回，
这样的话，虽然任务还没有执行完，但是程序可以继续执行下去，而不用等待这个函数的任务
在仿真中执行完。如果你想让程序在这里等待任务执行完，则只需要在后面加上.join()。
本例子就是让程序在这里等待无人机起飞任务完成，然后再执行降落任务。 
新的任务会打断上一个没有执行完的任务，所以如果takeoff函数没有加 .join()，
则最后的表现是无人机还没有起飞就降落了，无人机是不会起飞的。
"""
# 设置偏航
drivetrain = airsim.DrivetrainType.ForwardOnly # 永远朝向速度方向

# 获取估计状态
state = client.getMultirotorState()

# 获取状态真值
kinematics_state = client.simGetGroundTruthKinematics()

# take off
client.takeoffAsync().join()

# hover
client.hoverAsync().join()
time.sleep(2)

# move by position
client.moveToZAsync(-3, 2).join()  # 上升到3m高度
client.moveToPositionAsync(5, 0, -3, 2).join()  # 飞到（5,0）点坐标
client.moveToPositionAsync(5, 5, -3, 2).join()  # 飞到（5,5）点坐标
client.moveToPositionAsync(0, 5, -3, 2).join()  # 飞到（0,5）点坐标
client.moveToPositionAsync(0, 0, -3, 2).join()  # 回到（0,0）点坐标

client.hoverAsync().join()
time.sleep(2)
client.moveToZAsync(-10, 2).join()
client.hoverAsync().join()
time.sleep(2)

# move by velocity
client.moveByVelocityZAsync(8, 0, -10, 2).join()     # 以1m/s速度向前飞8秒钟
client.moveByVelocityZAsync(0, 8, -10, 2).join()     # 以1m/s速度向右飞8秒钟
client.moveByVelocityZAsync(-8, 0, -10, 2).join()    # 以1m/s速度向后飞8秒钟
client.moveByVelocityZAsync(0, -8, -10, 2).join()    # 以1m/s速度向左飞8秒钟

client.hoverAsync().join()
time.sleep(2)

# land
client.landAsync().join()

# lock
client.armDisarm(False)

# release control
client.enableApiControl(False)