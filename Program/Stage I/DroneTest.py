"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/7/7 21:47
__file__ = DroneTest.py
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

"""
STAGE I:
    We are going to create a group containing 3 drones, called Alpha, Bravo
and Charlie.
    Initially, the 3 drones will be put on a horizontal line with the same
interval.
    Alpha is always located in (0, 0, 0), in another word, the coordinate system 
is set according to the initial location of Alpha. 
    Bravo will be on the east, and Charlie will be on the west.
"""

# set the name
Alpha = ""
Bravo = "Bravo"
Charlie = "Charlie"

# connect to the AIRSIM
client = airsim.MultirotorClient()
# check the connection
client.confirmConnection()
print("Alpha created")
# unlock
client.enableApiControl(True, vehicle_name=Alpha)
if client.isApiControlEnabled(vehicle_name=Alpha):
    print("Alpha is under control\n")

# set the initial interval
interval = 15

# add the drones
bravo_start_pose = airsim.Pose(airsim.Vector3r(0, interval, 0), airsim.Vector3r(0, 0, 0))
if (
        client.simAddVehicle(Bravo, "simpleflight", bravo_start_pose)
):
    print("Bravo created")
    client.enableApiControl(True, vehicle_name=Bravo)
    if client.isApiControlEnabled(vehicle_name=Bravo):
        print("Bravo is under control\n")

charlie_start_pose = airsim.Pose(airsim.Vector3r(0, -interval, 0), airsim.Vector3r(0, 0, 0))
if (
        client.simAddVehicle(Charlie, "simpleflight", charlie_start_pose)
):
    print("Charlie created")
    client.enableApiControl(True, vehicle_name=Charlie)
    if client.isApiControlEnabled(vehicle_name=Charlie):
        print("Charlie is under control\n")

print("初始位置：\n"
      "\tAlpha(0, 0, 0)\n"
      "\tBravo(0, {}, 0)\n"
      "\tCharlie(0, {}, 0)\n".format(interval, -interval)
)


"""
STAGE II:
    We will first let the three drones take off and climb to the height of 3
meters, where they will hover for a while to keep steady.
    There are totally four kinds of transformations:
        -> v type (short for the 'vertical formation')
        -> t type (short for the 'triangular formation')
        -> l type (short for the 'linear formation(left)')
        -> r type (short for the 'linear formation(left)')
    We choose one of the transformations, set the goal, and start transforming.
"""

def Transform(type):
    para1 = random.randint(10, 31)
    if type == 'v':
        alpha_goal = (2 * para1, 0, -3)
        bravo_goal = (para1, -interval, -3)
        charlie_goal = (3 * para1, interval, -3)

    elif type == 't':
        alpha_goal = (2 * para1, 0, -3)
        bravo_goal = ((2 - (3**0.5) / 2) * para1, -interval + 0.5 * para1, -3)
        charlie_goal = ((2 - (3**0.5) / 2) * para1, interval - 0.5 * para1, -3)

    elif type == 'l':
        para2 = random.randint(10, 31)
        bravo_goal = (
            random.randint(10, 31),
            -interval + random.randint(10, 61),
            -3
        )
        alpha_goal = (
            bravo_goal[0] + para1,
            bravo_goal[1] + interval - para2,
            -3
        )
        charlie_goal = (
            alpha_goal[0] +para1,
            alpha_goal[1] + interval - para2,
            -3
        )

    elif type == 'r':
        para2 = random.randint(10, 31)
        charlie_goal = (
            random.randint(10, 31),
            interval - random.randint(10, 61),
            -3
        )
        alpha_goal = (
            charlie_goal[0] + para1,
            charlie_goal[1] - interval + para2,
            -3
        )
        bravo_goal = (
            alpha_goal[0] +para1,
            alpha_goal[1] - interval + para2,
            -3
        )

    else:
        print("Type error")
        sys.exit(-1)

    print("目标位置：\n"
          "\tAlpha {}\n"
          "\tBravo ({}, {}, {})\n"
          "\tCharlie ({}, {}, {})\n".format(
        alpha_goal, bravo_goal[0], bravo_goal[1] + interval, bravo_goal[2],
        charlie_goal[0], charlie_goal[1] - interval, charlie_goal[2]
        )
    )

    client.moveToPositionAsync(alpha_goal[0], alpha_goal[1], alpha_goal[2], 5, vehicle_name=Alpha)
    client.moveToPositionAsync(bravo_goal[0], bravo_goal[1], bravo_goal[2], 5, vehicle_name=Bravo)
    client.moveToPositionAsync(charlie_goal[0], charlie_goal[1], charlie_goal[2], 5, vehicle_name=Charlie)

for i in range(80):
    # take off (lift to 3m above the ground)
    client.takeoffAsync(vehicle_name=Alpha)
    client.takeoffAsync(vehicle_name=Bravo)
    client.takeoffAsync(vehicle_name=Charlie).join()

    client.moveToZAsync(-3, 1, vehicle_name=Alpha)
    client.moveToZAsync(-3, 1, vehicle_name=Bravo)
    client.moveToZAsync(-3, 1, vehicle_name=Charlie).join()

    client.hoverAsync(vehicle_name=Alpha)
    client.hoverAsync(vehicle_name=Bravo)
    client.hoverAsync(vehicle_name=Charlie)
    time.sleep(3)

    tag = i % 4
    if tag == 0:
        Transform('v')
    elif tag == 1:
        Transform('t')
    elif tag == 2:
        Transform('l')
    elif tag == 3:
        Transform('r')


    """
    STAGE III:
        We record the state of the drones, including their velocity and distances
    between one and another, and write down what we need in the file.
        After the transformation is over, we reset the simulator and go back to 
    STAGE II.
    """

    time.sleep(0.1)

    alpha_state = client.simGetGroundTruthKinematics(vehicle_name=Alpha)
    bravo_state = client.simGetGroundTruthKinematics(vehicle_name=Bravo)
    charlie_state = client.simGetGroundTruthKinematics(vehicle_name=Charlie)

    Avx, Avy, Avz, Bvx, Bvy, Bvz, Cvx, Cvy, Cvz = [], [], [], [], [], [], [], [], []
    Mabx, Maby, Mabz, Mbcx, Mbcy, Mbcz, Macx, Macy, Macz = [], [], [], [], [], [], [], [], []
    Dab, Dbc, Dac = [], [], []

    while (
        alpha_state.linear_velocity.get_length() > 0.1 or
        bravo_state.linear_velocity.get_length() > 0.1 or
        charlie_state.linear_velocity.get_length() > 0.1
    ):
        alpha_pos = alpha_state.position
        bravo_pos = airsim.Vector3r(
            bravo_state.position.x_val,
            bravo_state.position.y_val + interval,
            bravo_state.position.z_val
        )
        charlie_pos = airsim.Vector3r(
            charlie_state.position.x_val,
            charlie_state.position.y_val - interval,
            charlie_state.position.z_val
        )

        D = {
            "alpha's vx": alpha_state.linear_velocity.x_val,
            "alpha's vy": alpha_state.linear_velocity.y_val,
            "alpha's vz": alpha_state.linear_velocity.z_val,
            "bravo's vx": bravo_state.linear_velocity.x_val,
            "bravo's vy": bravo_state.linear_velocity.y_val,
            "bravo's vz": bravo_state.linear_velocity.z_val,
            "charlie's vx": charlie_state.linear_velocity.x_val,
            "charlie's vy": charlie_state.linear_velocity.y_val,
            "charlie's vz": charlie_state.linear_velocity.z_val,
            "Dab": alpha_pos.distance_to(bravo_pos),
            "Dbc": bravo_pos.distance_to(charlie_pos),
            "Dac": alpha_pos.distance_to(charlie_pos),
            "Mabx": 0.5 * (alpha_pos.x_val + bravo_pos.x_val),
            "Maby": 0.5 * (alpha_pos.y_val + bravo_pos.y_val),
            "Mabz": 0.5 * (alpha_pos.z_val + bravo_pos.z_val),
            "Mbcx": 0.5 * (bravo_pos.x_val + charlie_pos.x_val),
            "Mbcy": 0.5 * (bravo_pos.y_val + charlie_pos.y_val),
            "Mbcz": 0.5 * (bravo_pos.z_val + charlie_pos.z_val),
            "Macx": 0.5 * (alpha_pos.x_val + charlie_pos.x_val),
            "Macy": 0.5 * (alpha_pos.y_val + charlie_pos.y_val),
            "Macz": 0.5 * (alpha_pos.z_val + charlie_pos.z_val),

        }

        Avx.append(D["alpha's vx"])
        Avy.append(D["alpha's vy"])
        Avz.append(D["alpha's vz"])
        Bvx.append(D["bravo's vx"])
        Bvy.append(D["bravo's vy"])
        Bvz.append(D["bravo's vz"])
        Cvx.append(D["charlie's vx"])
        Cvy.append(D["charlie's vy"])
        Cvz.append(D["charlie's vz"])
        Dab.append(D["Dab"])
        Dbc.append(D["Dbc"])
        Dac.append(D["Dac"])
        Mabx.append(D["Mabx"])
        Maby.append(D["Maby"])
        Mabz.append(D["Mabz"])
        Mbcx.append(D["Mbcx"])
        Mbcy.append(D["Mbcy"])
        Mbcz.append(D["Mbcz"])
        Macx.append(D["Macx"])
        Macy.append(D["Macy"])
        Macz.append(D["Macz"])

        print(D)

        alpha_state = client.simGetGroundTruthKinematics(vehicle_name="")
        bravo_state = client.simGetGroundTruthKinematics(vehicle_name="Bravo")
        charlie_state = client.simGetGroundTruthKinematics(vehicle_name="Charlie")

        time.sleep(0.2)

    DATA = np.array([
        Avx[0: len(Avx) // 3],
        Avy[0: len(Avy) // 3],
        Avz[0: len(Avz) // 3],
        Bvx[0: len(Bvx) // 3],
        Bvy[0: len(Bvy) // 3],
        Bvz[0: len(Bvz) // 3],
        Cvx[0: len(Cvx) // 3],
        Cvy[0: len(Cvy) // 3],
        Cvz[0: len(Cvz) // 3],
        Dab[0: len(Dab) // 3],
        Dbc[0: len(Dbc) // 3],
        Dac[0: len(Dac) // 3],
        Mabx[0: len(Dac) // 3],
        Maby[0: len(Dac) // 3],
        Mabz[0: len(Dac) // 3],
        Mbcx[0: len(Dac) // 3],
        Mbcy[0: len(Dac) // 3],
        Mbcz[0: len(Dac) // 3],
        Macx[0: len(Dac) // 3],
        Macy[0: len(Dac) // 3],
        Macz[0: len(Dac) // 3],
    ])
    data = np.average(DATA, axis=1)

    with open("data_airsim.csv", 'a') as f:
        for d in data:
            f.write(str(d) + ",")
        f.write(str(tag) + '\n')
        print("已写入数据")
    print(data)
    print("进度：{}/80".format(i + 1))

    client.reset()
    client.enableApiControl(True, vehicle_name=Alpha)
    client.enableApiControl(True, vehicle_name=Bravo)
    client.enableApiControl(True, vehicle_name=Charlie)

