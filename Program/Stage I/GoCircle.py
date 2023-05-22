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

import airsim
import numpy as np
import math
import time

client = airsim.MultirotorClient()  # connect to the AirSim simulator
client2 = airsim.MultirotorClient()
client.enableApiControl(True)       # 获取控制权
client.armDisarm(True)              # 解锁
client2.enableApiControl(True)       # 获取控制权
client2.armDisarm(True)              # 解锁

client.takeoffAsync().join()        # 起飞
client.takeoffAsync().join()
client.moveToZAsync(-3, 1).join()   # 第二阶段：上升到2米高度

center = np.array([[0], [5]])    # 圆心设置
speed = 2                        # 速度设置
radius = 5                       # 半径设置
clock_wise = True                # 顺时针或逆时针设置

pos_reserve = np.array([[0.], [0.], [-3.]])

# 速度控制
for i in range(2000):
    # 获取无人机当前位置
    state = client.simGetGroundTruthKinematics()
    pos = np.array([[state.position.x_val], [state.position.y_val], [state.position.z_val]])

    # 计算径向速度的方向向量
    dp = pos[0:2] - center
    if np.linalg.norm(dp) - radius > 0.1:
        vel_dir_1 = -dp
    elif np.linalg.norm(dp) - radius < 0.1:
        vel_dir_1 = dp

    # 计算切向速度的方向向量
    theta = math.atan2(dp[1, 0], dp[0, 0])
    if clock_wise:
        theta += math.pi / 2
    else:
        theta -= math.pi / 2
    v_dir_2 = np.array([[math.cos(theta)], [math.sin(theta)]])

    # 计算最终速度的方向向量
    # 两个向量加权平均，k值是需要调节的，这样曲线才圆润
    v_dir = 0.08 * vel_dir_1 + v_dir_2
    # 计算最终速度指令(单位向量乘速度)
    v_cmd = speed * v_dir/np.linalg.norm(v_dir)
    # 速度控制
    client.moveByVelocityZAsync(v_cmd[0, 0], v_cmd[1, 0], -3, 1)

    # 画图
    point_reserve = [airsim.Vector3r(pos_reserve[0, 0], pos_reserve[1, 0], pos_reserve[2, 0])]
    point = [airsim.Vector3r(pos[0, 0], pos[1, 0], pos[2, 0])]
    point_end = pos + np.vstack((v_cmd, np.array([[0]])))
    point_end = [airsim.Vector3r(point_end[0, 0], point_end[1, 0], point_end[2, 0])]
    client.simPlotArrows(point, point_end, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0])
    client.simPlotLineList(point_reserve+point, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)

    # 循环
    pos_reserve = pos
    time.sleep(0.02)