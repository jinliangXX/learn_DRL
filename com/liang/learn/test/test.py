# -*- coding: UTF-8 -*-
"""
测试
"""
from com.liang.learn.utils.process_mnist import \
    get_training_data_set
import numpy as np
# import gym
#
# env = gym.make('MountainCar-v0')  # 定义使用 gym 库中的那一个环境
# env = env.unwrapped  # 不做这个会有很多限制


def getMaxSubstring(str):
    lenth = len(str)
    before = 0
    maxlen = 0
    cur = 0
    for i in range(lenth - 1):
        k = i - before - 1
        if before > 1 and k >= 0 and str[i] == str[k]:
            cur = before + 2
        elif i > 1 and str[i - 2] == str[i]:
            cur = 3
        elif i > 0 and str[i - 1] == str[i]:
            cur = 2
        else:
            cur = 1
        if cur > maxlen:
            maxlen = cur
        before = cur
    return maxlen


if __name__ == '__main__':
    s = input()
    print(s)
    maxs = getMaxSubstring(s)
    print(maxs)

    # if __name__ == '__main__':
    '''input_num = 10
    print(range(input_num))
    print([0.0 for _ in range(input_num)])'''
    '''a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    c = zip(a, b)
    print(list(c))'''
    # # a,b = get_training_data_set()
    # a = np.array([0, 1, 2])
    # d = 1 - a
    # print(a.shape)
    # b = np.array([9, 8, 7])
    # print(b.shape)
    # b.shape = (3, 1)
    # print(b.shape)
    # c = a + b
    # print(c.shape)
    # a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # for i in a[::-1]:
    #     print i
    # pass
    # print(env.action_space)  # 查看这个环境中可用的 action 有多少个
    # print(env.observation_space)  # 查看这个环境中可用的 state 的 observation 有多少个
    # print(env.observation_space.high)  # 查看 observation 最高取值
    # print(env.observation_space.low)  # 查看 observation 最低取值
    # from numpy import *
    #
    # num = 0
    # random.seed(5)
    # while (num < 5):
    #     print(random.random())
    #     num += 1
