"""
DDPG
Deep Deterministic Policy Gradient
"""

import tensorflow as tf
import numpy as np
import gym
import time

# 超参数
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
RENDER = False
ENV_NAME = 'Pendulum-v0'


# DDPG
class DDPG(object):
    '''
    DDPG的实现
    '''

    def __init__(self, a_dim, s_dim, a_bound, ):
        # 初始化memory
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1),
                               dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)

        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # 网络的参数
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Critic/target')

        # 更新target网络的参数
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea),
                              tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in
                             zip(self.at_params, self.ae_params, self.ct_params,
                                 self.ce_params)]

        q_target = self.R + GAMMA * q_

        # 计算td_error
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error,
                                                            var_list=self.ce_params)
        # 最大化q
        a_loss = -tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss,
                                                            var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        '''
        选择动作
        :param s: state
        :return:
        '''
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        '''
        学习
        :return:
        '''
        # target网络更新参数
        self.sess.run(self.soft_replace)

        # 随机选择memory
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain,
                      {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        '''
        存储
        :param s:
        :param a:
        :param r:
        :param s_:
        :return:
        '''
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        '''
        建立action中的网络
        :param s: 输入的状态
        :param scope: 名字
        :param trainable: 是否可训练
        :return:
        '''
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1',
                                  trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh,
                                name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        '''

        :param s:
        :param a:
        :param scope:
        :param trainable:
        :return:
        '''
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1],
                                   trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1],
                                   trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


# 训练
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration

t1 = time.time()

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995
            ddpg.learn()

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward),
                  'Explore: %.2f' % var, )
            break
    print('Running time: ', time.time() - t1)
