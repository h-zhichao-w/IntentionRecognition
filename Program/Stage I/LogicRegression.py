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

""""
现在需要写一个程序来判断每个人的分数是否及格，计分标准为：
总分=40%数学+30%语文+30%英语。总分大于等于60为及格，其余为不及格。
我想要得到的只有两个结果，及格或者不及格。我们可以简单理解为-1和1。
那我们怎么把总分的结果映射到-1和1上呢？这就需要使用一个特殊的函数了，
我们把这个函数叫做激活函数。
这就是逻辑回归，先通过一个线性模型得到一个结果，然后再通过激活函数将结果映射到指定范围。
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
# 准备x的数据
x = np.array([
    [60],
    [20],
    [30],
    [80],
    [59],
    [90]
])
# 准备y的数据
y = np.array([1, 0, 0, 1, 0, 1])
# 创建逻辑回归模型
lr = LogisticRegression()
# 填充数据并训练
lr.fit(x, y)
# 准备测试集
x_test = np.array([
    [62],
    [87],
    [39],
    [48]
])
# 判断测试数据是否及格
y_predict = lr.predict(x_test)
print(y_predict)