"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/8/5 23:30
__file__ = ovm.py
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_path = 'data_without_distance.csv'
label_path = 'label1.csv'

# Read data and labels
dataset = pd.read_csv(data_path)
labels = pd.read_csv(label_path)

# Create the model
Label1 = labels.iloc[ : , 1]
Label2 = labels.iloc[ : , 2]
Label3 = labels.iloc[ : , 3]
Data = dataset.iloc[ : ,  : ]

X1, x1, Y1, y1 = train_test_split(Data, Label1, test_size = 1/5, random_state = 0)

X_train = np.array(X1)
print(X_train, X_train.shape)
X_test = np.array(x1)
print(X_test, X_test.shape)
Y_train = np.array(Y1)
for i in range(Y_train.size):
    if Y_train[i] == -1:
        Y_train[i] = 0
Y_train = Y_train.reshape([Y_train.size, 1])
print(Y_train, Y_train.shape)
Y_test = np.array(y1)
for i in range(Y_test.size):
    if Y_test[i] == -1:
        Y_test[i] = 0
Y_test = Y_test.reshape([Y_test.size, 1])
print(Y_test, Y_test.shape)

model1 = keras.Sequential(
    [
        layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
)
model1.summary()

# 二分类问题
model1.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model1.fit(X_train, Y_train, batch_size= 64, epochs= 10)
score1 = model1.evaluate(X_test, Y_test, batch_size= 32)

X2, x2, Y2, y2 = train_test_split(Data, Label2, test_size = 1/5, random_state = 0)

X_train = np.array(X2)
print(X_train, X_train.shape)
X_test = np.array(x2)
print(X_test, X_test.shape)
Y_train = np.array(Y2)
for i in range(Y_train.size):
    if Y_train[i] == -1:
        Y_train[i] = 0
Y_train = Y_train.reshape([Y_train.size, 1])
print(Y_train, Y_train.shape)
Y_test = np.array(y2)
for i in range(Y_test.size):
    if Y_test[i] == -1:
        Y_test[i] = 0
Y_test = Y_test.reshape([Y_test.size, 1])
print(Y_test, Y_test.shape)

model2 = keras.Sequential(
    [
        layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
)
model2.summary()

# 二分类问题
model2.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model2.fit(X_train, Y_train, batch_size= 64, epochs= 10)
score2 = model2.evaluate(X_test, Y_test, batch_size= 32)

X3, x3, Y3, y3 = train_test_split(Data, Label3, test_size = 1/5, random_state = 0)

X_train = np.array(X3)
print(X_train, X_train.shape)
X_test = np.array(x3)
print(X_test, X_test.shape)
Y_train = np.array(Y3)
for i in range(Y_train.size):
    if Y_train[i] == -1:
        Y_train[i] = 0
Y_train = Y_train.reshape([Y_train.size, 1])
print(Y_train, Y_train.shape)
Y_test = np.array(y3)
for i in range(Y_test.size):
    if Y_test[i] == -1:
        Y_test[i] = 0
Y_test = Y_test.reshape([Y_test.size, 1])
print(Y_test, Y_test.shape)

model3 = keras.Sequential(
    [
        layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
)
model3.summary()

# 二分类问题
model3.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model3.fit(X_train, Y_train, batch_size= 64, epochs= 10)
score3 = model3.evaluate(X_test, Y_test, batch_size= 32)