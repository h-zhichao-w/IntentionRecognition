"""
__title__    = model_test.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2022/1/22 21:20
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

model = keras.models.load_model('model-drone0-20220130.h5')

data_path = 'Drone0-20220208.csv'

# Read data and labels
dataset = pd.read_csv(data_path)

# Create the model
Data = dataset.iloc[ : ,  : ]

X = np.array(Data)

for i in range(1, X.shape[0]):
     X_test = np.array([np.mean(np.array(Data[ : i + 1]), axis=0)])
     # print(X_test)
     Y = model.predict(X_test)
     plt.plot(i, Y[0, 0], 'or')
     plt.plot(i, Y[0, 1], 'oy')
     plt.plot(i, Y[0, 2], 'og')

plt.show()
