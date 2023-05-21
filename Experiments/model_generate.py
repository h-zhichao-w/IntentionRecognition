"""
__title__    = model_generate.py
__author__   = 'Hansen Wong // 王之超'
__time__     = 2022/1/22 20:44
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
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

data_path = 'model4.0\TRAIN-DATA-{}.csv'.format(time.strftime("%Y%m%d", time.localtime()))
label_path = 'model4.0\TRAIN-LABEL-{}.csv'.format(time.strftime("%Y%m%d", time.localtime()))

# Read data and labels
dataset = pd.read_csv(data_path)
labels = pd.read_csv(label_path)

# Create the model
Label = labels.iloc[ : , 0]
Data = dataset.iloc[ : ,  : ]

X_train, X_test, Y_train, Y_test = train_test_split(Data, Label, test_size = 1/5, random_state = 0)

X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

OHE_train = keras.utils.to_categorical(Y_train, num_classes=3)
OHE_test = keras.utils.to_categorical(Y_test, num_classes=3)

model = keras.Sequential(
    [
        layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ]
)
model.summary()

# 多分类问题的compile
opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
# keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, OHE_train, epochs= 20, batch_size= 10)
model.evaluate(X_test, OHE_test, batch_size= 10)

model.save('model4.0/model-{}.h5'.format(time.strftime("%Y%m%d", time.localtime())))
