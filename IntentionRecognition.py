"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/8/1 12:30
__file__ = IntentionRecognition.py
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
This file is intended to serve for the project, Research on Recognition of 
Aircraft Group Behavior Intention. This file will serve as a supporting pack
for the following project files.
"""

import airsim
import numpy as np
import matplotlib.pyplot as plt
import time

class SimulationPlatform:
    def __init__(self, ct: airsim.MultirotorClient, origin: np.ndarray, curse_speed: float, kp: float, HQ: airsim.Vector3r = airsim.Vector3r(), drone_number: int = 3):
        """
        :param ct: get the airsim client, so as to use the functions provided by airsim
        :param origin: the origin position of drones under PlayStart coordinate system
        :param curse_speed: the cursing speed of drones, and the unit is m/s
        :param HQ: coordinate information of HQ under PlayStart coordinate system
        """
        self.ct = ct
        self.origin = origin
        self.type = 'Drone'
        self.curse_speed = curse_speed
        self.HQ = HQ
        self.DroneNumber = drone_number
        self.Drone = []
        self.run = False
        self.r_min = 1.4
        if kp == 0.1:
            self.intention = 'CONTRACTION'
        elif kp == 0.:
            self.intention = 'FREESTYLE'
        else:
            self.intention = 'EXPANSION'

    def All_Armed_N_API(self):
        """
        combine the enableApiControl and armDisarm
        """
        for i in range(self.DroneNumber):
            self.ct.enableApiControl(True, vehicle_name=self.type + str(i))
            self.ct.armDisarm(True, vehicle_name=self.type + str(i))

    def Get_Drone_Pos(self, vehicle_index: int):
        """
        The function 'simGetGroundTruthKinematics' will return the position information, in
        the sef-coordinate system, however. In order to unify the coordinate system, use this
        function to transfer the position information from sef-coordinate system to the PlayStart
        coordinate system.

        :return: the position under PlayStart coordinate system
        """
        SelfPos = self.ct.simGetGroundTruthKinematics(vehicle_name=self.type + str(vehicle_index)).position
        x = SelfPos.x_val
        y = SelfPos.y_val
        z = SelfPos.z_val
        x += self.origin[vehicle_index][0]
        y += self.origin[vehicle_index][1]
        pos = airsim.Vector3r(x, y, z)
        return pos

    def All_Take_Off(self, height):
        """
        order all drones to take off at the same time and reach a given height
        :param height: in the NED coordinate system, if you want to lift the drone, you have to input a negative number
        """
        for i in range(self.DroneNumber):
            if i != self.DroneNumber - 1:
                self.ct.takeoffAsync(vehicle_name=self.type + str(i))
            else: self.ct.takeoffAsync(vehicle_name=self.type + str(i)).join()
        time.sleep(1)
        for i in range(self.DroneNumber):
            if i != self.DroneNumber - 1:
                self.ct.moveToZAsync(height, 2, vehicle_name=self.type + str(i))
            else: self.ct.moveToZAsync(height, 2, vehicle_name=self.type + str(i)).join()
        time.sleep(1.5)

    def Cal_Goal(self, height, theta: float, R: float):
        """
        :param theta: degree, north is zero, and increases along the inverse clockwise direction
        :param height: in the NED coordinate system, if you want to lift the drone, you have to input a negative number
        """
        goals = []
        for i in range(self.DroneNumber):
            x = 60 + np.cos(theta * 2 * np.pi / 360) * R
            y = -np.sin(theta * 2 * np.pi / 360) * R
            z = height
            goal = airsim.Vector3r(x, y, z)
            goals.append(goal)
            theta += 120
        self.goals = np.array(goals)

    def Goal_Transfer(self, goal: airsim.Vector3r, vehicle_index: int):
        """
        just quite opposite to Get_Drone_Pos
        """
        x = goal.x_val - self.origin[vehicle_index][0]
        y = goal.y_val - self.origin[vehicle_index][1]
        z = goal.z_val
        ngoal = airsim.Vector3r(x, y, z)
        return ngoal

    def Avoid_Collision(self):
        """
        To avoid the collision between drones
        """
        for i in range(self.DroneNumber):
            drone = self.type + str(i)
            pos = self.Get_Drone_Pos(i)
            rest = pos.distance_to(self.goals[i])
            for j in range(self.DroneNumber):
                if i != j:
                    neighbour = self.type + str(j)
                    pos_n = self.Get_Drone_Pos(j)
                    rest_n = pos_n.distance_to(self.goals[j])
                    rij = pos.distance_to(pos_n)
                    if rij < self.r_min:
                        if rest > rest_n:
                            print('Collision warning! Drone{} has to slow down!'.format(j))
                            self.ct.moveByVelocityAsync(0, 0, 0, 1.5, vehicle_name=neighbour).join()
                            self.ct.moveToPositionAsync(
                                self.Goal_Transfer(self.goals[j], j).x_val,
                                self.Goal_Transfer(self.goals[j], j).y_val,
                                self.Goal_Transfer(self.goals[j], j).z_val,
                                self.curse_speed, vehicle_name=neighbour
                            )
                        else:
                            print('Collision warning! Drone{} has to slow down!'.format(i))
                            self.ct.moveByVelocityAsync(0, 0, 0, 1.5, vehicle_name=drone).join()
                            self.ct.moveToPositionAsync(
                                self.Goal_Transfer(self.goals[i], i).x_val,
                                self.Goal_Transfer(self.goals[i], i).y_val,
                                self.Goal_Transfer(self.goals[i], i).z_val,
                                self.curse_speed, vehicle_name=drone
                            )

    def Check_Error(self):
        """
        check if the error is too large and the data is not reasonable
        :return:
        """
        for i in range(self.DroneNumber):
            if self.Get_Drone_Pos(i).distance_to(self.goals[i]) > 1.4:
                return False
        return True

    def Record(self):
        """
        record the data when the condition is correct
        save the state in the form of
        [linear_v_x, linear_v_y, linear_v_z, angular_v_x, angular_v_y, angular_v_z, linear_a_x, linear_a_y, linear_a_z, angular_a_x, angular_a_y, angular_a_z, orientation_x, orientation_y, orientation_z, orientation_w, R, angle]
        NONE * 18
        """
        able_to_detect = [False, False, False]
        is_move = [True, True, True]
        drones = [[], [], []]
        data = []
        time_st = time.time()
        self.operating_normally = True
        self.endpoints = [[], [], []]

        while(self.run):
            self.Avoid_Collision()

            for i in range(self.DroneNumber):
                pos = self.Get_Drone_Pos(i)
                state = self.ct.simGetGroundTruthKinematics(vehicle_name=self.type + str(i))
                time_nw = time.time()

                if is_move[i]:
                    if pos.distance_to(self.HQ) > 45:
                        continue
                    else:
                        able_to_detect[i] = True

                    if pos.distance_to(self.goals[i]) < 1 and state.linear_velocity.get_length() <= 0.75 * self.curse_speed:
                        is_move[i] = False
                        self.endpoints[i].append(pos)
                        s = (state.linear_velocity.get_length())**2 / (2 * state.linear_acceleration.get_length())
                        s_vec = state.linear_velocity.__truediv__(state.linear_velocity.get_length()).__mul__(s)
                        self.endpoints[i].append(pos.__add__(s_vec))


                if is_move[i] and able_to_detect[i]:
                    drones[i].append(
                        [state.linear_velocity.x_val, state.linear_velocity.y_val, state.linear_velocity.z_val, state.angular_velocity.x_val, state.angular_velocity.y_val, state.angular_velocity.z_val, state.linear_acceleration.x_val, state.linear_acceleration.y_val, state.linear_acceleration.z_val, state.angular_acceleration.x_val, state.angular_acceleration.y_val, state.angular_acceleration.z_val, state.orientation.x_val, state.orientation.y_val, state.orientation.z_val, state.orientation.w_val, pos.distance_to(self.HQ), np.arctan((pos.y_val - self.HQ.y_val) / (pos.x_val - self.HQ.x_val))]
                    )

                if time_nw - time_st > 40:
                    print('Error: too long operation time')
                    self.run = False
                    self.operating_normally = False
                    break
                else:
                    print('running time: {}'.format(time_nw - time_st))

            if (not is_move[0]) and (not is_move[1]) and (not is_move[2]):
                self.run = False
                break

        for i in range(self.DroneNumber):
            data.append(np.array(drones[i][ : int(len(drones[i]) / 2)]))
            print(data[i].shape)
            data[i] = np.mean(data[i], axis=0)
            print(data[i].shape)
            print(data[i])

        for i in range(self.DroneNumber - 1):
            if i == 0:
                self.data = np.hstack((data[0], data[1]))
            else:
                self.data = np.hstack((self.data, data[i + 1]))
        print(self.data.shape)

    def Move_2_Goal_Together(self):
        """
        move to respective goal together
        """
        ngoals = self.goals.copy()
        for i in range(self.DroneNumber):
            if i != 0:
                ngoals[i] = self.Goal_Transfer(ngoals[i], i)
        for i in range(self.DroneNumber):
            self.ct.moveToPositionAsync(ngoals[i].x_val, ngoals[i].y_val, ngoals[i].z_val, self.curse_speed, vehicle_name=self.type + str(i))
        self.run = True
        self.Record()

    def Move_On_Path_Together(self, route: list):
        for i in range(self.DroneNumber):
            if i != 0:
                for j in range(len(route[i])):
                    route[i][j] = self.Goal_Transfer(route[i][j], i)

        for i in range(self.DroneNumber):
            self.ct.moveOnPathAsync(route[i], self.curse_speed, vehicle_name=self.type + str(i))

        self.run = True
        is_move = [True, True, True]        # to indicate respectively that the drone is still moving
        drones = [[], [], []]               # real-time data storage space
        data = []                           # eventual data storage space
        time_st = time.time()               # the time that start to move
        self.operating_normally = True      # the movement is still normal
        self.endpoints = [[], [], []]       # storage space for endpoints

        while self.run:
            # storage space for the positions of the drones under the start point coordinate system
            pos = []
            # storage space for the states of the drones
            states = []
            # current time
            time_nw = time.time()
            for i in range(self.DroneNumber):
                pos.append(self.Get_Drone_Pos(i))
                states.append(self.ct.simGetGroundTruthKinematics(vehicle_name=self.type + str(i)))
            # real-time center
            centre = airsim.Vector3r(
                (pos[0].x_val + pos[1].x_val + pos[2].x_val) / 3.,
                (pos[0].y_val + pos[1].y_val + pos[2].y_val) / 3.,
                (pos[0].z_val + pos[1].z_val + pos[2].z_val) / 3.
            )

            for i in range(self.DroneNumber):
                if is_move[i]:
                    if states[i].position.distance_to(route[i][-1]) < 1. and states[i].linear_velocity.get_length() <= 0.75 * self.curse_speed:
                        is_move[i] = False
                        self.endpoints[i].append(pos[i])
                        s = (states[i].linear_velocity.get_length()) ** 2 / (2 * states[i].linear_acceleration.get_length())
                        s_vec = states[i].linear_velocity.__truediv__(states[i].linear_velocity.get_length()).__mul__(s)
                        self.endpoints[i].append(pos[i].__add__(s_vec))
                        print(states[i].position.distance_to(route[i][-1]), states[i].linear_velocity.get_length())

                    drones[i].append(
                        [states[i].linear_velocity.x_val, states[i].linear_velocity.y_val, states[i].linear_velocity.z_val,
                         states[i].angular_velocity.x_val, states[i].angular_velocity.y_val, states[i].angular_velocity.z_val,
                         states[i].linear_acceleration.x_val, states[i].linear_acceleration.y_val,
                         states[i].linear_acceleration.z_val, states[i].angular_acceleration.x_val,
                         states[i].angular_acceleration.y_val, states[i].angular_acceleration.z_val, states[i].orientation.x_val,
                         states[i].orientation.y_val, states[i].orientation.z_val, states[i].orientation.w_val,
                         pos[i].distance_to(centre), pos[i].distance_to(pos[0]), pos[i].distance_to(pos[1]), pos[i].distance_to(pos[2]),
                         np.arctan2(centre.y_val - pos[i].y_val, centre.x_val - pos[i].x_val)
                        ]
                    )

                if time_nw - time_st > 30:
                    print('Error: too long operation time')
                    self.run = False
                    self.operating_normally = False
                    break
                else:
                    print('running time: {}'.format(time_nw - time_st))

            if (not is_move[0]) and (not is_move[1]) and (not is_move[2]):
                self.run = False
                break

        for i in range(self.DroneNumber):
            data.append(np.array(drones[i][: int(len(drones[i]) / 2)]))
            print(data[i].shape)
            data[i] = np.mean(data[i], axis=0)
            print(data[i].shape)
            print(data[i])

        for i in range(self.DroneNumber - 1):
            if i == 0:
                self.data = np.hstack((data[0], data[1]))
            else:
                self.data = np.hstack((self.data, data[i + 1]))
        print(self.data.shape)

    def Check_Intention(self):
        """
        to identify the intention by calculating the distance between the goal, or the end point, to the HQ
        """
        intention = {'contraction': 0, 'freestyle': 0, 'expansion': 0}
        for i in range(self.DroneNumber):
            if self.HQ.distance_to(self.endpoints[i][1]) <= 15:
                intention['contraction'] += 1
            elif self.HQ.distance_to(self.endpoints[i][1]) <= 30:
                intention['freestyle'] += 1
            else:
                intention['expansion'] += 1
        if intention['contraction'] >= 2:
            self.intention = 'CONTRACTION'
        elif intention['freestyle'] >= 2:
            self.intention = 'FREESTYLE'
        else:
            self.intention = 'EXPANSION'

    def Draw_Circle(self, center: tuple, radius, type: str):
        """
        to draw a circle
        :param type: the kind you like the circle to be
        """
        x = np.linspace(center[0] - radius, center[0] + radius, 2000)
        y_plus = (radius ** 2 - (x - center[0]) ** 2) ** 0.5 + center[1]
        y_minus = -(radius ** 2 - (x - center[0]) ** 2) ** 0.5 + center[1]
        plt.plot(x, y_plus, type)
        plt.plot(x, y_minus, type)

    def Plot_Routine(self):
        """
        watch out that, you have to exchange x and y due to the NED coordinate system
        """
        plt.title(self.intention)
        plt.xlabel('East')
        plt.ylabel('North')
        plt.grid(True)
        self.Draw_Circle((0, 60), 15, 'r--')
        self.Draw_Circle((0, 60), 30, 'y--')
        self.Draw_Circle((0, 60), 45, 'g--')
        plt.plot(np.array([self.HQ.y_val]), np.array([self.HQ.x_val]), 's')
        for i in range(self.DroneNumber):
            plt.scatter(self.origin[i][1], self.origin[i][0], c='r')
            plt.scatter(self.goals[i].y_val, self.goals[i].x_val, c='m')
            plt.scatter(self.endpoints[i][0].y_val, self.endpoints[i][0].x_val, c='b')
            plt.text(0, 50 - i * 5, self.type + str(i) + ' %.2f'%(self.endpoints[i][0].distance_to(self.goals[i])))
            plt.scatter(self.endpoints[i][1].y_val, self.endpoints[i][1].x_val, c='k')
            plt.arrow(
                x=self.origin[i][1],
                y=self.origin[i][0],
                dx=self.endpoints[i][0].y_val - self.origin[i][1],
                dy=self.endpoints[i][0].x_val - self.origin[i][0],
                head_width=0.1,
                head_length=0.2,
                length_includes_head=True,
            )
            plt.arrow(
                x=self.endpoints[i][0].y_val,
                y=self.endpoints[i][0].x_val,
                dx=self.endpoints[i][1].y_val - self.endpoints[i][0].y_val,
                dy=self.endpoints[i][1].x_val - self.endpoints[i][0].x_val,
                head_width=0.1,
                head_length=0.2,
                length_includes_head=True
            )

        plt.show()

    def write(self, d_name: str, l_name: str):
        """
        write down the data and the label if the error is not too large
        """
        if self.operating_normally:
            with open(d_name, 'a') as f:
                for i in self.data:
                    f.write(str(i))
                    f.write(',')
                f.write('\n')

            with open(l_name, 'a') as l:
                if self.intention == 'CONTRACTION':
                    l.write('0' + ',' + '1' + ',' + '0' + ',' + '0')
                elif self.intention == 'FREESTYLE':
                    l.write('1' + ',' + '0' + ',' + '1' + ',' + '0')
                else:
                    l.write('2' + ',' + '0' + ',' + '0' + ',' + '1')
                l.write('\n')
        else:
            print('Too Large Error(s)')


