import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)

# 超参数
OUTPUT_GRAPH = False
# 最大回合数
MAX_EPISODE = 3000
# 展示的reward极限？阈值
DISPLAY_REWARD_THRESHOLD = 200
# 每一回合执行最大的步数
MAX_EP_STEPS = 1000
# 是否呈现浪费的时间
RENDER = False
# TD error（比平时好多少）
GAMMA = 0.9
# actor的learning rate
LR_A = 0.001
# critic的learning rate
LR_C = 0.01

# 获取游戏的环境
env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

# observation的可选数量
N_F = env.observation_space.shape[0]
# action的可选数量
N_A = env.action_space.n


class Actor(object):
    def __init__(self, sess, n_features, n_actions,
                 lr=0.001):
        '''
        actor初始化方法
        :param sess:
        :param n_features:
        :param n_actions:
        :param lr: learning_rate
        '''
        self.sess = sess
        # state
        self.s = tf.placeholder(tf.float32, [1, n_features],
                                "state")
        # action
        self.a = tf.placeholder(tf.int32, None, "act")
        # TD error
        self.td_error = tf.placeholder(tf.float32, None,
                                       "td_error")

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),
                bias_initializer=tf.constant_initializer(
                    0.1),
                name='l1'
            )
            self.acts_prob = tf.layers.dense(
                inputs=11,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(
                    0., .1),
                bias_initializer=tf.constant_initializer(
                    0.1),
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            # log action概率
            log_prob = tf.log(self.acts_prob[0, self.a])
            # 通过TD error求解loss     log 概率 * TD 方向
            self.exp_v = tf.reduce_mean(
                log_prob * self.td_error)

        with tf.variable_scope('train'):
            # minimize(-exp_v) = maximize(exp_v)
            self.train_op = tf.train.AdamOptimizer(
                lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a,
                     self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        '''
        选择动作
        :param s: state，即observation
        :return:
        '''
        s = s[np.newaxis, :]
        # 神经网络输出值
        probs = self.sess.run(self.acts_prob, {s: self.s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        '''
        初始化方法
        :param sess: session
        :param n_features: state
        :param lr: learning ratemm
        '''
        self.sess = sess
        # state
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        # ?
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        # reward
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                # weights
                kernel_initializer=tf.random_normal_initializer(0., .1),
                # bias
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )
