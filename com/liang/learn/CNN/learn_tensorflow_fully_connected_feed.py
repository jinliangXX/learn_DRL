# -*- coding: UTF-8 -*-
import argparse
import logging
import math
import os
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data, \
    mnist

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10


def inference(images, hidden1_units, hidden2_units):
    '''

    :param images: 输入
    :param hidden1_units: 第一个隐藏层的大小
    :param hidden2_units: 第二个隐藏层的大小
    :return: 输出tensor with computed logits
    '''
    # 隐藏层1
    try:
        with tf.name_scope('hidden1'):
            weights = tf.Variable(
                tf.truncated_normal(
                    [IMAGE_PIXELS, hidden1_units],
                    stddev=1.0 / math.sqrt(
                        float(IMAGE_PIXELS))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden1_units]),
                                 name='biases')
            hidden1 = tf.nn.relu(
                tf.matmul(images, weights) + biases)
    except:
        logging.error("创建第一个隐藏层失败", exc_info=True)
    # 隐藏层2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal(
            [hidden1_units, hidden2_units],
            stddev=1.0 / math.sqrt(
                float(hidden2_units))), name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(
            tf.matmul(hidden1, weights) + biases)
    # 隐藏层3
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden2_units, NUM_CLASSES],
                stddev=1.0 / math.sqrt(
                    (float(hidden2_units))),
                name='weights'))
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        # 最终输出
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def get_loss(logits, labels):
    '''
    计算损失函数
    :param logits: inference的输出
    :param labels:
    :return:
    '''
    labels = tf.to_int64(labels)
    return tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)


def training(loss, learning_rate):
    '''
    训练模型
    :param loss:
    :param learning_rate:
    :return:
    '''
    tf.summary.scalar('loss', loss)
    # 使用随机梯度下降的算法
    optimizer = tf.train.GradientDescentOptimizer(
        FLAGS.learning_rate)
    # 创建一个保存全局训练步骤的值
    global_step = tf.Variable(0, name='global_step',
                              trainable=False)
    # 更新权重，最小化loss
    train_op = optimizer.minimize(loss,
                                  global_step=global_step)
    return train_op


def evaluation(logits, labels):
    '''
    用于判断预测值与标签之间是否相同
    :param logits:
    :param labels:
    :return: 预测正确的样本数量
    '''
    # 判断是否正确
    correct = tf.nn.in_top_k(logits, labels, 1)
    # 将正确的汇总转换输出
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def placeholder_inputs(batch_size):
    '''
    定义输入与占位符
    :param batch_size:
    :return:
    '''
    images_placeholder = tf.placeholder(tf.float32, shape=(
        batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int64,
                                        shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    '''
    生成训练数据
    :param data_set:
    :param images_pl:
    :param labels_pl:
    :return:
    '''
    images_feed, labels_feed = data_set.next_batch(
        FLAGS.batch_size,
        FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    '''
    对所有回合的数据进行评估
    :param sess:
    :param eval_correct:
    :param images_placeholder:
    :param labels_placeholder:
    :param data_set:
    :return:
    '''
    # 预测正确的数量
    true_count = 0
    # steps_per_epoch = data_set.num_example // FLAGS.batch_size
    steps_per_epoch = FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct,
                               feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print(
        '全部数量: %d  预测正确的数量: %d  百分比 @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
    data_sets = input_data.read_data_sets(
        FLAGS.input_data_dir, FLAGS.fake_data)
    with tf.Graph().as_default():
        # 获取占位符
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)
        # 获取整个模型
        logits = inference(images_placeholder,
                           FLAGS.hidden1, FLAGS.hidden2)
        # 获取loss
        loss = get_loss(logits, labels_placeholder)
        # 获取训练模型的方法
        train_op = training(loss, FLAGS.learning_rate)
        # 获取预测正确的样本数量
        eval_correct = evaluation(logits,
                                  labels_placeholder)
        # 获取tensor的总和
        summary = tf.summary.merge_all()
        # 创建检查点文件
        saver = tf.train.Saver()
        # 创建会话
        sess = tf.Session()
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir, sess.graph)
        # 初始化会话中的变量
        init = tf.global_variables_initializer()
        sess.run(init)
        # 开始循环训练
        for step in range(FLAGS.max_steps):
            # 当前时间
            start_time = time.time()
            # 获取训练值
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                # 输出loss
                print("Step %d: loss = %.2f (%.3f sec)" % (
                    step, loss_value, duration))
                # 更新事件文件
                summary_str = sess.run(summary,
                                       feed_dict=feed_dict)
                summary_writer.add_summary(summary_str,
                                           step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (
                    step + 1) == FLAGS.max_steps:
                # 保存参数
                checkpoint_file = os.path.join(
                    FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file,
                           global_step=step)
                # 对训练集进行评估
                print('训练集评估:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                print('验证集评估:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                print('测试集评估:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    # 设置超参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    # parser.add_argument(
    #     '--input_data_dir',
    #     type=str,
    #     default=os.path.join(
    #         os.getenv('TEST_TMPDIR', '/Users/xujinliang/DRL/project/learn_DRL'),
    #         'com/liang/learn/CNN/MNIST_data'),
    #     help='Directory to put the input data.'
    # )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join(
            os.getenv('TEST_TMPDIR',
                      'G:\DRL\project\learn_DRL'),
            'com\liang\learn\CNN\MNIST_data'),
        help='Directory to put the input data.'
    )
    # parser.add_argument(
    #     '--log_dir',
    #     type=str,
    #     default=os.path.join(
    #         os.getenv('TEST_TMPDIR', '/Users/xujinliang/DRL/project/learn_DRL'),
    #         'com/liang/learn/log'),
    #     help='Directory to put the log data.'
    # )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(
            os.getenv('TEST_TMPDIR',
                      'G:\DRL\project\learn_DRL'),
            'com\liang\learn\log'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
