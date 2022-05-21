"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/9/2 17:08
__file__ = ann_train.py
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

model = keras.models.load_model('ModelName.h5')

data_path = 'data.csv'
label_path = 'label.csv'

# Read data and labels
dataset = pd.read_csv(data_path)
labels = pd.read_csv(label_path)

# Create the model
Label = labels.iloc[ : , 0]
Data = dataset.iloc[ : ,  : ]

X_train, X_test, Y_train, Y_test = train_test_split(Data, Label, test_size = 1/1.1, random_state = 0)

X_train = np.array(X_train)
X_test = np.array(X_test)

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

OHE_train = keras.utils.to_categorical(Y_train, num_classes=3)
OHE_test = keras.utils.to_categorical(Y_test, num_classes=3)

model.evaluate(X_test, OHE_test)

model.fit(X_train, OHE_train, batch_size= 1, epochs= 10)

model.evaluate(X_test, OHE_test)

model.save('ModelName.h5')
