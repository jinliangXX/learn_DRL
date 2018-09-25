# 目录结构 #

	├── README.md
	├── com
	│   └── liang
	│       └── learn
	│           ├── CNN        #CNN网络学习
	│           │   ├── cnn_tensorflow.py        #TensorFlow实现CNN网络，并训练cifar
	│           │   ├── full_connect_object.py        #用面向对象的方法实现全连接网络
	│           │   ├── full_connect_tensorflow.py        #使用TensorFlow实现全连接网络
	│           │   ├── full_connect_vector.py        #使用向量的方法实现全连接网络
	│           │   ├── learn_tensorflow_fully_connected_feed.py        #使用标准的TensorFlow方法实现全连接，并训练mnist数据
	│           │   ├── linear_unit_main.py        #基于感知器实现线性单元
	│           │   ├── perception_main.py        #训练感知器，即perceptron.py
	│           │   └── perceptron.py        #实现感知器
	│           ├── DQN
	│           │   ├── QLearningTable.py      #实现Q learning中的table
	│           │   ├── Q_learning.py      #实现最简单的Q learning
	│           │   ├── RL_brain.py      #环境的反馈类
	│           │   ├── double_DQN_learning.py      #double DQN的实现
	│           │   ├── dqn_learning.py      #学习、实现DQN
	│           │   ├── dqn_learning_CNN.py      #尝试使用CNN取代q table
	│           │   └── policy_gradient.py      #Policy Gradients的学习、实现
	│           ├── LSTM        LSTM网络学习
	│           │   ├── LSTM_learn.py
	│           │   └── RNN_learn.py
	│           ├── log        #训练网络打印的日志
	│           ├── test        #测试
	│           │   ├── full_connect.py
	│           │   └── test.py
	│           └── utils        #工具类，一般包含下载、读取训练数据等
	└── tree.md



# 工作记录 #
## 2018.09.24 ##
1. 继续学习TensorFlow源码中对RNN的实现，主要是对PTB数据集的训练(LSTM/LSTM_learn.py)
2.

## 2018.09.23 ##
1. 感冒头疼，休息一天   T.T


## 2018.09.22 ##
1. 学习重现TensorFlow源码中对RNN的实现，主要是对PTB数据集的训练(LSTM/LSTM_learn.py)
2. 了解RNN/LSTM有关的方法后，继续完成识别MNIST数据集的任务
3. 阅读TensorFlow对RNN/LSTM封装的函数，与RNN/LSTM的原理相对照验证学习

## 2018.09.21 ##
1. 学习循环神经网络RNN的相关知识(https://zybuluo.com/hanbingtao/note/541458)
2. 学习LSTM型循环神经网络相关知识(https://zybuluo.com/hanbingtao/note/581764)
3. 尝试使用TensorFlow编写RNN程序识别mnist数据集，但是TensorFlow专门为RNN包装了较多方法和类，未果~

## 2018.09.20 ##
1. 尝试使用CNN取代DQN中的神经网络，由于使用gym的游戏简单，维度太低，不适用。
2. 学习Policy Gradient算法，并对比DQN论文了解其使用
3. 对Policy Gradient算法进行简单实现，完成木棒游戏(cDQN/policy_gradient.py)

## 2018.09.19 ##
1. 编码过程中对TensorFlow了解不够深，重新系统学习可视化、数据读取、多线程等等(http://www.tensorfly.cn/tfdoc/how_tos/overview.html)
2. 学习DRL中的OpenAI gym环境库的使用
3. 学习DQN相关算法Sarsa，对比其与DQN的不同

## 2018.09.18 ##
1. 了解DQN的相关理论，并与Q learning对比学习
2. 完成使用简单神经网络的DQN，并自动学习走迷宫游戏(DQN/dqn_learning.py)
3. 在简单DQN上增加Double DQN和experience replay机制(DQN/dqn_learning.py)

## 2018.09.17 ##
1. 学习Q learning的相关知识及概念
2. 实现简单的Q learning强化学习网络，使之自己玩游戏
3. 使用Q learning完成走迷宫的游戏(DQN/Q_learning.py)

## 2018.09.16 ##
1. 学习TensorFlow中对于CNN神经网络封装的函数，例如cconv2d,maxpooling等等
2. 调试cnn_tensorflow.py程序，使之跑通，完成训练及测试(CNN/cnn_tensorflow.py)
3. 学习RL中的相关概念及分类

## 2018.09.15 ##
1. 复习卷积神经网络(https://www.zybuluo.com/hanbingtao/note/485480)
2. 学习使用tensorflow构建CNN神经网络，使之训练CIFAR-10数据集(CNN/cnn_tensorflow.py)
3. 学习TensorFlow中可视化的内容

## 2018.09.14 ##
1. 学习TensorFlow基本概念与使用方法(http://www.tensorfly.cn/)
2. 使用TensorFlow实现全连接神经网络(CNN/full_connect_tensorflow.py)
3. 阅读并实现TensorFlow源码中examples/fully_connected_feed.py的功能


## 2018.09.13 ##
1. 使用向量的方法实现全连接神经网络(CNN/full_connect_vector.py)
2. 使用全神经网络训练mnist
3. 调试训练过程

## 2018.09.12 ##
1. 复习线性单元+监督学习&无监督学习+目标函数
2. 复习梯度下降优化算法，以及BGD/SGD
3. 在感知器的基础上实现线性单元，并进行训练(CNN/linear_unit_main.py)
4. 复习神经网络+反向传播算法+梯度检查
5. 使用面向对象的方法实现全连接神经网络(CNN/full_connect_object.py)


##  2018.09.11  ##
1. 复习深度学习感知器的内容
2. 手动实现感知器，不使用其他框架(CNN/perceptron.py)
3. 使用监督学习训练感知器，使之实现and函数(CNN/perception_main.py)