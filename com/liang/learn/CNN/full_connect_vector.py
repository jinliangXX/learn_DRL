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
