"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/7/20 16:17
__file__ = IntentionData_Test.py
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
import time
import random
import sys
import matplotlib.pyplot as plt

# 出生点坐标系下的起始位置
origin_pos = np.array([
    [0, 0],
    [90, 51.96],
    [90, -51.96]
])
# HQ的坐标
HQ = airsim.Vector3r(60, 0, -8.5)
# 巡航速度
curse_speed = 15
# 既定高度
height = -9

# 坐标系的变换
def get_Drone_pos(client: airsim.MultirotorClient(), vehicle_name: str):
    state = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
    x = state.position.x_val
    y = state.position.y_val
    z = state.position.z_val
    index = int(vehicle_name[-1])
    x += origin_pos[index][0]
    y += origin_pos[index][1]
    pos = airsim.Vector3r(x, y, z)
    return pos

def goal_transfer(x, y, z, vehicle_name: str):
    index = int(vehicle_name[-1])
    x -= origin_pos[index][0]
    y -= origin_pos[index][1]
    goal = airsim.Vector3r(x, y, z)
    return goal

def cal_goal(
        index: int,
        R = random.randint(0, 4500) / 100.0,
        theta = random.randint(0, 3600) / 10.0
):
    """
        在NED坐标系下，x轴正方向（即正北）为theta = 0，逆时针方向theta变大
    """
    x = 60 + np.cos(theta * 2 * np.pi / 360) * R
    y = -np.sin(theta * 2 * np.pi / 360) * R
    if index == 0:
        return airsim.Vector3r(x, y, height), theta
    else:
        return airsim.Vector3r(x, y, height)

def check_intention(goal_list, total_num = 3):
    intention = {'contraction': 0, 'freestyle': 0, 'expansion': 0}
    for i in range(total_num):
        if HQ.distance_to(goal_list[i]) <= 15:
            intention['contraction'] += 1
        elif HQ.distance_to(goal_list[i]) <= 30:
            intention['freestyle'] += 1
        else:
            intention['expansion'] += 1
    if intention['contraction'] >= 2:
        print('intention: contraction')
    elif intention['freestyle'] >= 2:
        print('intention: freestyle')
    else: print('intention: expansion')

def draw_circle(center: tuple, radius, type: str):
    x = np.linspace(center[0]-radius, center[0]+radius, 2000)
    y_plus = (radius**2 - (x - center[0])**2)**0.5 + center[1]
    y_minus = -(radius**2 - (x - center[0])**2)**0.5 + center[1]
    plt.plot(x, y_plus, type)
    plt.plot(x, y_minus, type)

def plot_routine(origin, goal):
    """
        注意，我们已知的起点和终点都是基于NED坐标系，但是matplotlib绘画是右手坐标系
        因此，我们需要将x，y互换
    """
    plt.title('Routine')
    plt.xlabel('East')
    plt.ylabel('North')
    plt.grid(True)
    draw_circle((0, 60), 15, 'r--')
    draw_circle((0, 60), 30, 'y--')
    draw_circle((0, 60), 45, 'g--')
    for i in range(3):
        plt.scatter(origin[i][1], origin[i][0], c='r')
        plt.scatter(goal[i].y_val, goal[i].x_val, c='m')
        plt.plot(np.array([origin[i][1], goal[i].y_val]), np.array([origin[i][0], goal[i].x_val]), 'b-')
        # plt.plot(pt1, pt2, 'b-')
    plt.show()

def SetPath(ct: airsim.MultirotorClient, index: int, goal: airsim.Vector3r):
    if index == 0:
        g = goal
        pos = ct.simGetGroundTruthKinematics(vehicle_name='Drone'+str(index)).position
        path = airsim.Vector3r(
            g.x_val - pos.x_val,
            g.y_val - pos.y_val,
            g.z_val - pos.z_val
        )
    else:
        g = goal_transfer(goal.x_val, goal.y_val, goal.z_val, vehicle_name='Drone'+str(index))
        pos = ct.simGetGroundTruthKinematics(vehicle_name='Drone'+str(index)).position
        path = airsim.Vector3r(
            g.x_val - pos.x_val,
            g.y_val - pos.y_val,
            g.z_val - pos.z_val
        )

    return path


ct = airsim.MultirotorClient()
ct.confirmConnection()

# 获取控制权，解锁
for i in range(3):
    ct.enableApiControl(True, vehicle_name='Drone'+str(i))
    ct.armDisarm(True, vehicle_name='Drone'+str(i))

# 起飞
for i in range(3):
    if i != 2:
        ct.takeoffAsync(vehicle_name='Drone'+str(i))
    else:
        ct.takeoffAsync(vehicle_name='Drone'+str(i)).join()

# 获得高度
for i in range(3):
    if i != 2:
        ct.moveToZAsync(height, 2, vehicle_name='Drone'+str(i))
    else:
        ct.moveToZAsync(height, 2, vehicle_name='Drone'+str(i)).join()
time.sleep(1)

# 设置终点，保存在列表goal中，为出生点坐标系，后期调用注意转换
goal = [0]
theta = 0
for i in range(3):
    if i == 0:
        goal[0], theta = cal_goal(i)
    else:
        theta += 120
        goal.append(cal_goal(i, theta=theta))
goal = np.array(goal)
print(goal)

check_intention(goal)

plot_routine(origin_pos, goal)

for i in range(3):
    if i != 2:
        ct.moveToPositionAsync(goal[i].x_val, goal[i].y_val, goal[i].z_val, curse_speed, vehicle_name='Drone'+str(i))
    else:
        ct.moveToPositionAsync(goal[i].x_val, goal[i].y_val, goal[i].z_val, curse_speed, vehicle_name='Drone'+str(i)).join()