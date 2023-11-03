训练数据由两张excel表提供

node表:[节点， 类型， 名]
relation表:[节点， 节点， 关系]

表的路径在parameters.py中定义
训练模型的控制参数也在此文件定义

运行顺序:
    1.运行random_walk.py 生成test train valid数据集
    2.运行main.py 训练模型，模型保存于./saved_model/下
    3.运行recommendation.py 得到推荐论文