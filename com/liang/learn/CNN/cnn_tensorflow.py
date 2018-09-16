# -*- coding: UTF-8 -*-
import argparse

import tensorflow as tf
import sys

from com.liang.learn.utils.model_pruning.examples.cifar10 import cifar10_pruning

"""
Created by xujinliang on 2018.9.15
"""


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # 获取训练数据和标签
        images,labels = cifar10_pruning.distorted_inputs()

        #


def main(argv=None):
    cifar10_pruning.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/Users/xujinliang/DRL/project/learn_DRL/com/liang/learn/train_dir/cifar',
        help='Directory where to write event logs and checkpoint.')
    parser.add_argument(
        '--pruning_hparams',
        type=str,
        default='',
        help="""Comma separated list of pruning-related hyperparameters""")
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000000,
        help='Number of batches to run.')
    parser.add_argument(
        '--log_device_placement',
        type=bool,
        default=False,
        help='Whether to log device placement.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
