# -*- coding: UTF-8 -*-
import numpy as np


class FullConnectedLayer(object):
    '''
    全连接层实现类
    '''

    def __init__(self, input_size, output_size, activator):
        '''
        构造函数
        :param input_size:本层输入向量的维度
        :param output_size:本层输出向量的维度
        :param activator:激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W,产生均匀分布的数组
        self.W = np.random.uniform(-0.1, 0.1, (
            output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        前向计算
        :param input_array:输入向量，维度必须等于input_size
        :return:
        '''
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        :param delta_array:
        :return:
        '''
        # 误差项
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        # 梯度
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grab = delta_array

    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        :param learning_rate:
        :return:
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad


class SigmoidActivator(object):
    '''
    Sigmoid激活函数类
    '''

    def forward(self, weighted_input):
        '''
        正向计算
        :param weighted_input:
        :return:
        '''
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        '''
        求导数
        :param output:
        :return:
        '''
        return output * (1 - output)


class Network(object):
    '''
    神经网络类
    '''

    def __init__(self, layers):
        '''
        构造函数
        :param layers:
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(layers[i], layers[i + 1],
                                   SigmoidActivator()))

    def predict(self, sample):
        '''
        使用神经网络实现预测
        :param sample:输入样本
        :return:
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        :param labels: 样本标签
        :param data_set: 输入样本
        :param rate: 学习速率
        :param epoch: 训练轮数
        :return:
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        '''
        训练单个样本
        :param label:
        :param sample:
        :param rate:
        :return:
        '''
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        '''
        计算梯度
        :param label:
        :return:
        '''
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (
                label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        '''
        更新权重
        :param rate:
        :return:
        '''
        for layer in self.layers:
            layer.update(rate)
