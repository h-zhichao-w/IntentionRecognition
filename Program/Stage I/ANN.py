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

"""
神经网络是建立在逻辑回归之上的，可以说神经网络就是一个逻辑回归的集合。
机器学习中的神经元就是逻辑回归。神经网络通常有成百上千个参数，我们会
得到一个非常复杂的模型。虽然不能理解参数的含义，但是这些参数通常会给
我们一个很好的结果。不过这也正是神经网络的神奇之处。

神经网络通常会有三个部分，输入层由我们的特征数决定。而输出层由我们分
类数量决定，而中间部分为隐藏层：

使用scikit-learn，我们可以很快搭建一个神经网络。接下来我们用
scikit-learn中自带的数据集来实现一个神经网络
"""

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
# 加载数据集
iris_data = load_iris()
# 拆分数据集
x_train, x_test, y_train, y_test = train_test_split(
    iris_data['data'], iris_data['target'], test_size=0.25, random_state=1
)
# 创建神经网络模型
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[6, 2], random_state=0)
# 填充数据并训练
mlp.fit(x_train, y_train)
# 评估模型
score = mlp.score(x_test, y_test)
print(score)

"""
这里我们使用的是scikit-learn自带的鸢尾花数据，我们用train_test_split
将数据集分割成了两个部分，分别是训练集的特征和目标值，以及测试集的特征和目标值。

然后我们创建MLPClassifier类的实例，实际上它就是一个用于分类的多重感知机。
我们只需要关注hidden_layer_sizes参数即可，它就是我们神经网络的层数和节点数。
因为是一个二分类问题，所以这里的输出层有两个节点。
"""