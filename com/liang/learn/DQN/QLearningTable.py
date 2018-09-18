# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np


class QLearningTable:
    # 初始化
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9,
                 e_greedy=0.9):
        '''
        初始化方法
        :param actions: 动作
        :param learning_rate: 学习率
        :param reward_decay: 奖励衰减
        :param e_greedy: 贪婪度
        '''
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        '''
        选择行为
        :param observation: 现在的状态
        :return:
        '''
        # 检测state是否存在q_table中
        self.check_state_exits(observation)

        # 选择action
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]

            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)

        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        '''
        更新q_table
        :param s: state
        :param a: action
        :param r: reward
        :param s_: state_
        :return:
        '''
        # 检测state是否存在于q_table
        self.check_state_exits(s_)
        # 预测
        q_predict = self.q_table.loc[s, a]
        # 如果下步没有结束
        if s_ != 'terminal':
            # r + q预测
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        # q = q + lr*(r + q预测 — q)
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exits(self, state):
        '''
        检测state是否存在
        :param state:
        :return:
        '''
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns,
                          name=state))
