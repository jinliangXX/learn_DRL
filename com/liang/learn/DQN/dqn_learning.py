# -*- coding: UTF-8 -*-

from com.liang.learn.utils.maze_env import Maze
import tensorflow as tf


class DeepQNetwork:
    def _build_net(self):
        pass

    def inference(self, inputs):
        '''
        构建神经网络
        :param inputs:
        :return:
        '''
        # eval_net网络
        with tf.name_scope('eval_net'):
            with tf.name_scope('l1'):
                weights = tf.Variable()

            pass


if __name__ == '__main__':
    env = Maze()
