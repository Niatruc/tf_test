import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data # 使用tensorflow提供的手写数字图片数据集

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义一个添加神经层的函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))  #随机变量矩阵
            tf.summary.histogram(layer_name + '/Weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 列表， 全部是0.1
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases  #计算出预测值（未被激活）
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784], name='x_input') #28*28=784
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input')

# 增加一层输出层(用softmax做多分类)
prediction = add_layer(xs, 784, 10, 1, activation_function=tf.nn.softmax) # 隐藏层10个神经元

# 预测值和实际值之间的误差
with tf.name_scope('loss_using_cross_entropy'):
    # 用交叉熵函数
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-10,100.0)), reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)
    # tf.summary.histogram()
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy) # 0.1是学习效率(learning rate)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
writer = tf.summary.FileWriter("logs/", sess.graph)
summaries = tf.summary.merge_all()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0: # 每50步打印一次误差
        print(compute_accuracy(mnist.test.images, mnist.test.labels)) # 注意这里使用的是测试集数据

    sum = sess.run(summaries, feed_dict={xs: batch_xs, ys: batch_ys})
    writer.add_summary(sum, global_step=i)


print("over")