"""
__title__    = DataTest.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2022/2/16 10:52
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
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import NumericalAnalysis as na
from matplotlib.widgets import Button

folder = 'standard/'

# for i in range(3):
#      rd_path = folder+'Drone{}-plus-20220216.csv'.format(i) # data path
#      wt_path = folder+'Drone{}-omega-20220216-0.0.csv'.format(i)
#      D = pd.read_csv(rd_path)
#      data = np.array(D.iloc[1 : , 2])
#      length = len(data)
#      t = np.zeros(length) # step data
#
#      for j in range(length):
#           t[j] = j * 1
#
#      omega = np.zeros(length - 1)
#      for j in range(length - 1):
#           ipt = na.piecewiselinear_interploation(t, data, t[i] + 0.001)
#           omega[j] = (ipt - data[j]) / 0.001           # 角速度 - 方向角插值差商
#
#      with open(wt_path, 'a') as f:
#           f.write('omega_{}\n'.format(i))
#           for w in omega:
#                f.write(str(w) + '\n')

model = keras.models.load_model('model2.0/model-20220130.h5')

rd_path = []
for i in range(3):
     rd_path.append(folder+'Drone{}-plus-20220216-0.05.csv'.format(i))
# for i in range(3):
#      rd_path.append(folder+'Drone{}-omega-20220216-0.0.csv'.format(i))

Data = []
r = []
for i in range(len(rd_path)):
     D = pd.read_csv(rd_path[i])
     if i < 3:
          Data.append(np.array(D.iloc[:, :2]))
     else:
          Data.append(np.array(D.iloc[:, :1]))
     r.append(Data[i].shape[0])

row = min(r)

if len(Data) > 3:
     Drone = []
     for i in range(3):
          Drone.append(np.hstack((Data[i][:row], Data[i+3][:row])))
     X = np.hstack((np.hstack((Drone[0], Drone[1])), Drone[2]))
else:
     X = np.hstack((np.hstack((Data[0][:row], Data[1][:row])), Data[2][:row]))


print(X)

C = np.zeros(X.shape[0] - 1)
F = np.zeros(X.shape[0] - 1)
E = np.zeros(X.shape[0] - 1)

for i in range(1, X.shape[0]):
     X_test = np.array([np.mean(X[ : i + 1], axis=0)])
     # print(X_test)
     p = model.predict(X_test)[0]
     C[i-1] = p[0]
     F[i-1] = p[1]
     E[i-1] = p[2]

step = np.arange(0, C.shape[0], 1)
plt.title('kp = 0.05')
plt.xlabel('Step')
plt.ylabel('Probability')
plt.ion()

for i in range(1, C.shape[0]):
     plt.plot(step[:i], C[:i], 'or', label='kp = 0.05')
     plt.plot(step[:i], F[:i], 'oy', label='kp = 0.0')
     plt.plot(step[:i], E[:i], 'og', label='kp = -0.05')
     plt.legend()
     # plt.show()
     plt.clf()
     plt.cla()

plt.ioff()
