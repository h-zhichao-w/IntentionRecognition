"""
__title__ = 'testLR.py'
__author__ = 'Hansen Wong // 王之超'
__mtime__ = 2019/3/28
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃            ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('data.csv')
Para = dataset.iloc[0: , 0: 15].values
Output = dataset.iloc[0: , 15].values

X_train, X_test, Y_train, Y_test = train_test_split(Para, Output, test_size = 1/5, random_state = 0)
train = LogisticRegression()
train.fit(X_train, Y_train)

Y_pre = train.predict(X_test)
correct = 0
for i in range(len(Y_pre)):
    print("参数{}，预测为{}，实际为{}，结果为{}".format(
        X_test[i], Y_pre[i], Y_test[i], Y_pre[i] == Y_test[i])
    )
    if (Y_pre[i] - Y_test[i] == 0):
        correct += 1
rate = correct / len(Y_pre) * 100
print("准确率为{}%".format(rate))