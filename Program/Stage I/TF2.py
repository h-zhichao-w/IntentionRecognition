"""
__title__ = 'TF2.py'
__author__ = 'Hansen Wong // 王之超'
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

import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data.csv')
Para = dataset.iloc[0: , 0: 15].values
Output = dataset.iloc[0: , 15].values

train_data, test_data, train_label, test_label = train_test_split(Para, Output, test_size = 1/5, random_state = 0)

train_data.shape = (train_data.shape[0], 1, 15)
test_data.shape = (test_data.shape[0], 1, 15)
print("训练集大小为", train_data.shape, "训练集标签", train_label.shape)
print("测试集大小为", test_data.shape, "测试集标签", test_label.shape)

# Build the model
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(1, 15)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
)

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_data, train_label, epochs=5)

# Evaluate the accuracy
test_loss, test_acc = model.evaluate(test_data, test_label)

print("Test Accuracy: ", test_acc)