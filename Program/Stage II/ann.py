"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/8/5 15:25
__file__ = ann.py
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
Label = labels.iloc[ : , 0]
Data = dataset.iloc[ : ,  : ]

X_train, X_test, Y_train, Y_test = train_test_split(Data, Label, test_size = 1/5, random_state = 0)

X_train = np.array(X_train)
print(X_train, X_train.shape)
X_test = np.array(X_test)
print(X_test, X_test.shape)
Y_train = np.array(Y_train)
print(Y_train, Y_train.shape)
Y_test = np.array(Y_test)
print(Y_test, Y_test.shape)

for i in range(Y_train.size):
    Y_train[i] -= 1
for i in range(Y_test.size):
    Y_test[i] -= 1

OHE_train = keras.utils.to_categorical(Y_train, num_classes=3)
print(OHE_train)
print(OHE_train.shape)
OHE_test = keras.utils.to_categorical(Y_test, num_classes=3)
print(OHE_test)
print(OHE_test.shape)

model = keras.Sequential(
    [
        layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
        layers.Dense(3, activation='softmax')
    ]
)
model.summary()

# 多分类问题的compile
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, OHE_train, epochs= 10, batch_size= 64)
score = model.evaluate(X_test, OHE_test, batch_size=32)