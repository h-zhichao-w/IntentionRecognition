"""
__title__    = Refaction_and_Test.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2022/2/28 0:12
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

class Standard:
    def __init__(self):
        self.folder = 'standard/'
        self.intentions = ['--0.02', '-0.05']
        self.data = []
        for i in range(3):
            for intention in self.intentions:
                D = pd.read_csv(
                    self.folder + 'Drone{}-plus-20220216{}.csv'.format(i, intention)
                )
                self.data.append(np.array(D.iloc[:, :2]))

class Data:
    def __init__(self, intention: float, date):
        self.folder = 'kp = {}/'.format(intention)
        self.data = []
        for i in range(3):
            D = pd.read_csv(
                self.folder + 'Drone{}-plus-{}.csv'.format(i, date)
            )
            self.data.append(np.array(D.iloc[:, :2]))

    def refraction(self, S: Standard):
        for i in range(3):
            r1 = S.data[i*2].shape[0] / self.data[i].shape[0]
            r2 = S.data[i*2 + 1].shape[0] / self.data[i].shape[0]
            for j in range(self.data[i].shape[0]):
                k = int(i * r1)
                l = int(i * r2)
                if abs(self.data[i][j][0] - S.data[i*2][k][0]) < abs(self.data[i][j][0] - S.data[i*2+1][l][0]):
                    self.data[i][j][0] = S.data[i*2][k][0]
                else:
                    self.data[i][j][0] = S.data[i*2+1][l][0]
                if abs(self.data[i][j][1] - S.data[i*2][k][1]) < abs(self.data[i][j][1] - S.data[i*2+1][l][1]):
                    self.data[i][j][1] = S.data[i * 2][k][1]
                else:
                    self.data[i][j][1] = S.data[i * 2 + 1][l][1]

    def recombine(self):
        r = 10000
        for d in self.data:
            if r > d.shape[0]:
                r = d.shape[0]
        self.X = np.hstack((np.hstack((self.data[0][:r], self.data[1][:r])),self.data[2][:r]))


standard = Standard()
data = Data(0.0, 20220223)
data.refraction(standard)
data.recombine()
print(data.X)

from tensorflow import keras
from matplotlib import pyplot as plt
model = keras.models.load_model('model4.0/model-20220228.h5')
for i in range(1, data.X.shape[0]):
     X_test = np.array([np.mean(data.X[ : i + 1], axis=0)])
     Y = model.predict(X_test)
     print(Y)
     if i == 1:
          plt.plot(i, Y[0, 0], 'or', label='kp = 0.05')
          plt.plot(i, Y[0, 1], 'oy', label='kp = 0.0')
          plt.plot(i, Y[0, 2], 'og', label='kp = -0.05')
     else:
          plt.plot(i, Y[0, 0], 'or')
          plt.plot(i, Y[0, 1], 'oy')
          plt.plot(i, Y[0, 2], 'og')
plt.title('kp = 0.05')
plt.xlabel('Step')
plt.ylabel('Probability')
plt.legend()
plt.show()




