# -*- coding: UTF-8 -*-

from com.liang.learn.utils.maze_env import Maze
import tensorflow as tf
import numpy as np


class DeepQNetwork:
    def __init__(
            self,
            # 输出多少个action的值
            n_actions,
            # 长宽高等，用feature预测action的值
            n_features,
            # 学习速率
            learning_rate=0.01,
            # reward的折扣值
            reward_decay=0.9,
            # 贪婪算法的值，代表90%的时候选择预测最大的值
            e_greedy=0.9,
            # 隔多少步更换target神经网络的参数变成最新的
            replace_target_iter=300,
            # 记忆库的容量大小
            memory_size=500,
            # 神经网络学习时一次学习的大小
            batch_size=32,
            # 不断的缩小随机的范围
            e_greedy_increment=None,
            output_graph=False,
            double_q=True,
            sess=None
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        # epsilon的最大值
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        # 记录学习了多少步
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros(
            (self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e
                                  in
                                  zip(t_params, e_params)]

        # self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        # self.sess.run(tf.global_variables_initializer())
        # 记录下每步的误差
        self.cost_his = []
        self.double_q = double_q
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        # TensorFlow申请占位符，为CNN的输入，即state
        self.s = tf.placeholder(tf.float32,
                                [None, self.n_features],
                                name='s')  # input
        # 申请占位符，Q_target
        self.q_target = tf.placeholder(tf.float32, [None,
                                                    self.n_actions],
                                       name='Q_target')  # for calculating loss
        # eval_net神经网络部分
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params',
                 tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0.,
                                             0.3), tf.constant_initializer(
                    0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            # eval_net神经网络第一层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features,
                                            n_l1],
                                     initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1],
                                     initializer=b_initializer,
                                     collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            # eval_net神经网络第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2',
                                     [n_l1, self.n_actions],
                                     initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable('b2',
                                     [1, self.n_actions],
                                     initializer=b_initializer,
                                     collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        # 求误差
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target,
                                      self.q_eval))
        # 梯度下降
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(
                self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        # 接收下个state
        self.s_ = tf.placeholder(tf.float32,
                                 [None, self.n_features],
                                 name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params',
                       tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features,
                                            n_l1],
                                     initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1],
                                     initializer=b_initializer,
                                     collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2',
                                     [n_l1, self.n_actions],
                                     initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable('b2',
                                     [1, self.n_actions],
                                     initializer=b_initializer,
                                     collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    # 存储记忆
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    # 选择行为
    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval,
                                          feed_dict={
                                              self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    # 学习
    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],
                # fixed params
                self.s: batch_memory[:, :self.n_features],
                # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size,
                                dtype=np.int32)
        eval_act_index = batch_memory[:,
                         self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[
            batch_index, eval_act_index] = reward + self.gamma * np.max(
            q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 c ause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # 查看学习效果
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)),
                 self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def run_maze():
    # 控制到第几步学习
    step = 0
    # 进行300回合游戏
    for episode in range(300):
        # initial observation
        # 初始化环境，相当于每回合重新开始
        observation = env.reset()

        while True:
            # fresh env
            # 刷新环境
            env.render()

            # RL choose action based on observation
            # RL通过观察选择应该选择的动作
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            # 环境根据动作，做出反应，主要给出state，reward
            observation_, reward, done = env.step(action)

            # DQN存储记忆，即（s,a,r,s）
            RL.store_transition(observation, action, reward,
                                observation_)

            # 当学习次数大于200，且是5的倍数时才让RL学习
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            # 将下一个state作为本次的state
            observation = observation_

            # break while loop when end of this episode
            # 如果游戏结束，则跳出循环
            if done:
                break
            # 学习的次数
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
