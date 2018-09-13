# -*- coding: UTF-8 -*-
import numpy as np
from datetime import datetime

from com.liang.learn.utils.process_mnist import \
    get_training_data_set, get_test_data_set


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
        now = np.dot(self.W, input_array)
        now.shape = self.b.shape
        # now_a = now + self.b
        self.output = self.activator.forward(now + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        :param delta_array:
        :return:
        '''
        # 误差项
        now = self.activator.backward(self.output)
        now_a = np.dot(self.W.T, delta_array)
        # self.delta = self.activator.backward(
        #     self.input) * np.dot(
        #     self.W.T, delta_array)
        self.delta = now * now_a
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
                self.train_one_sample(labels[d],
                                      data_set[d], rate)

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
        # now_a = self.layers[-1].activator.backward(
        #     self.layers[-1].output)
        # now_b = label - self.layers[-1].output
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output) * (
                        label - self.layers[-1].output)
        self.layers[-1].delta = delta
        i = True
        for layer in self.layers[::-1]:
            if i:
                i = False
                continue
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


def get_result(vec):
    '''
    获取网络的识别结果
    :param vec:
    :return:
    '''
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index


def evaluate(network, test_data_set, test_labels):
    '''
    获取网络的错误率
    :param network:
    :param test_data_set:
    :param test_labels:
    :return:
    '''
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(
            network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)


def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    # train_data_set, train_labels = get_training_data_set()
    train_data_set, train_labels = get_test_data_set()
    test_data_set, test_labels = get_test_data_set()
    network = Network([784, 300, 10])
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.3, 1)
        print '%s epoch %d finished' % (
            datetime.now(), epoch)
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data_set,
                                   test_labels)
            print '%s after epoch %d, error ratio is %f' % (
                datetime.now(), epoch, error_ratio)
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio


if __name__ == '__main__':
    train_and_evaluate()
