# -*- coding: UTF-8 -*-
import argparse
import re
import sys
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning

from com.liang.learn.utils.model_pruning.examples.cifar10 import \
    cifar10_pruning, cifar10_input

"""
Created by xujinliang on 2018.9.15
"""
TOWER_NAME = 'tower'
BATCH_SIZE = 128
NUM_CLASSES = cifar10_input.NUM_CLASSES
INITIAL_LEARNING_RATE = 0.1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.9999


def _variable_with_weight_decay(name, shape, stddev, wd):
    '''
    初始化卷积层的参数
    :param name:
    :param shape:
    :param stddev:
    :param wd:
    :return:
    '''
    dtype = tf.float32
    var = tf.get_variable(name, shape,
                          initializer=tf.truncated_normal_initializer(
                              stddev=stddev, dtype=dtype))
    # 是否递减
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                   name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images):
    '''
    构造模型
    :param images:
    :return:
    '''
    # conv1
    with tf.variable_scope('conv1') as scope:
        # 卷积参数
        kernel = _variable_with_weight_decay('weight', shape=[5, 5, 3, 64],
                                             stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(images, pruning.apply_mask(kernel, scope),
                            [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', conv1.op.name)
        tf.summary.histogram(tensor_name + '/activations', conv1)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(conv1))
    # pool1
    pool1 = tf.nn.max_pool(
        conv1,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weight', shape=[5, 5, 64, 64],
                                             stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(norm1, pruning.apply_mask(kernel, scope),
                            [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(name='biases', shape=[64],
                                 initializer=tf.constant_initializer(0.1),
                                 dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', conv2.op.name)
        tf.summary.histogram(tensor_name + '/activations', conv2)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(conv2))
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(
        norm2,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool2'
    )
    # local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = tf.get_variable('biases', [384],
                                 initializer=tf.constant_initializer(0.1),
                                 dtype=tf.float32)
        local3 = tf.nn.relu(
            tf.matmul(reshape, pruning.apply_mask(weights, scope)) + biases,
            name=scope.name)
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', local3.op.name)
        tf.summary.histogram(tensor_name + '/activations', local3)
        tf.summary.scalar(tensor_name + '/sparsity',
                          tf.nn.zero_fraction(local3))
    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = tf.get_variable(name='biases', shape=[192],
                                 initializer=tf.constant_initializer(0.1),
                                 dtype=tf.float32)
        local4 = tf.nn.relu(
            tf.matmul(local3, pruning.apply_mask(weights, scope)) + biases,
            name=scope.name
        )
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', local4.op.name)
        tf.summary.histogram(tensor_name + '/activations', local4)
        tf.summary.scalar(tensor_name + '/sparsity',
                          tf.nn.zero_fraction(local4))
    # softmax_linear
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=0.0)
        biases = tf.get_variable(name='biases', shape=[NUM_CLASSES],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        softmax_linear = tf.add(
            tf.matmul(local4, pruning.apply_mask(weights, scope)),
            biases,
            name=scope.name)
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '',
                             softmax_linear.op.name)
        tf.summary.histogram(tensor_name + '/activations', softmax_linear)
        tf.summary.scalar(tensor_name + '/sparsity',
                          tf.nn.zero_fraction(softmax_linear))

    return softmax_linear


def _loss(logits, labels):
    '''
    获取loss
    :param logits:
    :param labels:
    :return:
    '''
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    '''
    将loss加入到summary中
    :param total_loss:
    :return:
    '''
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def _train(total_loss, global_step):
    '''
    创建优化器并更新权值
    :param total_loss:
    :param global_step:
    :return:
    '''
    # 影响learning_rate的变量
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # 衰减learning_rate值
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True
    )
    tf.summary.scalar('learning_rate', lr)

    # 生成所有loss和相关总结的移动平行线
    loss_averages_op = _add_loss_summaries(total_loss)

    # 计算梯度
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # 申请梯度
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # 添加直方图（训练变量的）
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # 添加梯度的直方图
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                          global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # 获取训练数据和标签
        images, labels = cifar10_pruning.distorted_inputs()

        # 建立图
        logits = inference(images)

        # 获取loss
        loss = _loss(logits, labels)

        # 创建模型（更新等操作）
        train_op = _train(loss, global_step)

        # 解析超参数
        pruning_hparams = pruning.get_pruning_hparams().parse(
            FLAGS.pruning_hparams)

        pruning_obj = pruning.Pruning(pruning_hparams, global_step=global_step)

        mask_update_op = pruning_obj.conditional_mask_update_op()

        pruning_obj.add_pruning_summaries()

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = 128
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = (
                        '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print(format_str % (
                        datetime.now(), self._step, loss_value,
                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
                # Update the masks
                mon_sess.run(mask_update_op)


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
        default='train_dir/cifar',
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
