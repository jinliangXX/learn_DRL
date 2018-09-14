# -*- coding: UTF-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 导入数据集
mnist = input_data.read_data_sets('MNIST_data',
                                  one_hot=True)

# 定义一些参数
learning_rate = 0.1  # 学习参数
batch_size = 100  # 每次训练数量
num_steps = 500
display_step = 20

# 定义神经网络的各层
n_hidden = 300
num_input = 784
num_class = 10

# 定义x,y
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_class])

# 存储隐藏层的W,B
weights = {
    'h': tf.Variable(
        tf.random_normal([num_input, n_hidden])),
    'out': tf.Variable(
        tf.random_normal([n_hidden, num_class]))
}
biases = {
    'b': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([num_class]))
}


# 创建模型
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h']),
                     biases['b'])
    layer_out = tf.add(tf.matmul(layer_1, weights['out']),
                       biases['out'])
    return layer_out


# 构造模型
logits = neural_net(X)

# 定义损失函数,resuce_mean将每张图片相加
loss_op = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                            labels=Y))
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# 初始化变量
init = tf.global_variables_initializer()

# 训练模型
with tf.Session() as sess:
    sess.run(init)

    for step in range(1, num_steps + 1):
        avg_cost = 0
        # 计算全部数据分几次训练
        total_batch = int(
            mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            # 获取批量训练数据
            batch_x, batch_y = mnist.train.next_batch(
                batch_size)
            acc, loss = sess.run([train_op, loss_op],
                                 feed_dict={X: batch_x,
                                            Y: batch_y})
            # print("轮数:", '%04d' % (step + 1),
            #       "loss={:.9f}".format(loss))
            # 计算平均损失
            avg_cost += loss / total_batch

        if step % display_step == 0:
            print("轮数:", '%04d' % (step + 1),
                  "loss={:.9f}".format(avg_cost))
    print("训练结束！")

    # 测试模型
    pred = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(pred, 1),
                            tf.argmax(Y, 1))

    # 计算准确率
    a = tf.case(correct_pred, tf.float32)
    accuracy = tf.reduce_mean(
        tf.case(correct_pred, tf.float32))
    print("Accuracy:", sess.run(accuracy, feed_dict={
        X: mnist.test.images,
        Y: mnist.test.labels}))
