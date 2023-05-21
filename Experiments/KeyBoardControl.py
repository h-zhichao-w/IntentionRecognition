"""
__title__    = KeyBoardControl.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2022/2/10 22:04
__Software__ = Pycharm
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
import logging
import time
import cv2
from djitellopy import tello
import KeyPressModule as kp  # 用于获取键盘按键
from time import sleep
import numpy as np

def getKeyboardInput(drone, speed, image):
    lr, fb, ud, yv = 0, 0, 0, 0
    key_pressed = 0

    if kp.getKey("e"):
        cv2.imwrite('D:/snap-{}.jpg'.format(time.strftime("%H%M%S", time.localtime())), image)

    if kp.getKey("UP"):
        Drone.takeoff()
    elif kp.getKey("DOWN"):
        Drone.land()

    if kp.getKey("j"):
        key_pressed = 1
        lr = -speed
    elif kp.getKey("l"):
        key_pressed = 1
        lr = speed

    if kp.getKey("i"):
        key_pressed = 1
        fb = speed
    elif kp.getKey("k"):
        key_pressed = 1
        fb = -speed

    if kp.getKey("w"):
        key_pressed = 1
        ud = speed
    elif kp.getKey("s1"):
        key_pressed = 1
        ud = -speed

    if kp.getKey("a"):
        key_pressed = 1
        yv = -speed
    elif kp.getKey("d"):
        key_pressed = 1
        yv = speed

    InfoText = "battery : {0}% height: {1}cm  time: {2}  yaw:{3}".format(drone.get_battery(), drone.get_height(), time.strftime("%H:%M:%S",time.localtime()), drone.get_yaw())
    cv2.putText(image, InfoText, (10, 20), font, fontScale, (0, 0, 255), lineThickness)
    if key_pressed == 1:
        InfoText = "Command : lr:{0}cm/s1 fb:{1}cm/s1 ud:{2}cm/s1 yv:{3}cm/s1".format(lr, fb, ud, yv)
        cv2.putText(image, InfoText, (10, 40), font, fontScale, (0, 0, 255), lineThickness)
        InfoText = "Status : a_x:{0}cm/s1^2, a_y:{1}cm/s1^2".format(drone.get_acceleration_x(), drone.get_acceleration_y())
        cv2.putText(image, InfoText, (10, 60), font, fontScale, (0, 0, 255), lineThickness)

    drone.send_rc_control(lr, fb, ud, yv)

# 主程序
# 摄像头设置
Camera_Width = 720
Camera_Height = 480
DetectRange = [6000, 11000]  # DetectRange[0] 是保持静止的检测人脸面积阈值下限，DetectRange[0] 是保持静止的检测人脸面积阈值上限
PID_Parameter = [0.5, 0.0004, 0.4]
pErrorRotate, pErrorUp = 0, 0

# 字体设置
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255, 0, 0)
lineThickness = 1

# Tello初始化设置
Drone = tello.Tello()  # 创建飞行器对象
Drone.connect()  # 连接到飞行器
Drone.streamon()  # 开启视频传输
Drone.LOGGER.setLevel(logging.ERROR)  # 只显示错误信息
sleep(5)  #  等待视频初始化
kp.init()  # 初始化按键处理模块

while True:
    OriginalImage = Drone.get_frame_read().frame
    Image = cv2.resize(OriginalImage, (Camera_Width, Camera_Height))
    getKeyboardInput(drone=Drone, speed=20, image=Image)  # 按键控制
    cv2.imshow("Drone Control Centre", Image)
    cv2.waitKey(1)
