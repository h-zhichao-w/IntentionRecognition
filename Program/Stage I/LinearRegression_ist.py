"""
__title__ = ''
__author__ = 'Thompson'
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
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cmath

#准备X的数据，X的数据是多维的，一组X向量对应一个Y
X = np.array([
    [1], [2], [3], [4], [5]
])
#准备Y的数据
Y = np.array(
    [1, 3, 5, 7, 9]
)
#创建线性回归模块
lr = LinearRegression()
#填充数据并训练
lr.fit(X, Y)
#输出参数
print("w =", lr.coef_, "b =", lr.intercept_)
#使用predict预测后面的数据，注意X的数据是二维的
y_predict = lr.predict(np.array([[100]]))
print(y_predict)

"""
接下来做一个实战，文件夹里有一个tsensor.csv
我去做对里面lnR和1/K的数据做线性回归
"""

#读取数据
dataset = pd.read_csv("tsensor.csv")
#获取数据
lnR = dataset.iloc[1: , 8].values
K_1 = dataset.iloc[1: , 7].values
#iloc所得为一维数组，需要改变形状
K_1 = K_1.reshape(len(K_1), 1)
#将数据分为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(K_1, lnR, test_size = 1/4, random_state = 0)
train = LinearRegression()
#回归计算
train.fit(X_train, Y_train)
#预测
lnR_pre = train.predict(X_test)
#可视化分析，先将拟合结果打在公屏上
plt.scatter(X_train, Y_train, color='blue')
plt.plot(X_train, train.predict(X_train), color='red')
plt.show()
#再将预测结果打在公屏上
plt.scatter(X_test, Y_test, color='blue')
plt.plot(X_test, train.predict(X_test), color='red')
plt.show()