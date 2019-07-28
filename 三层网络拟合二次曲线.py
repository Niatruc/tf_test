import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug

# 定义一个添加神经层的函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'zlayer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))  #随机变量矩阵
            tf.summary.histogram('/Weights' + str(n_layer), Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 列表， 全部是0.1
            tf.summary.histogram('/biases' + str(n_layer), biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases  #计算出预测值（未被激活）
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram('/outputs' + str(n_layer), outputs)
        return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis] #[-1, 1]，300行
noise = np.random.normal(0, 0.05, x_data.shape) # 三参是数据格式
y_data = np.square(x_data) - noise # 平方运算

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input') #tf.float32一定要加上
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(xs, 1, 10, 1, activation_function=tf.nn.relu) # 隐藏层10个神经元
prediction = add_layer(l1, 10, 1, 2, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    # tf.summary.histogram()
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 0.1是学习效率(learning rate)

init = tf.initialize_all_variables()
sess = tf.Session()
sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')
writer = tf.summary.FileWriter("logs/", sess.graph)
summaries = tf.summary.merge_all()
sess.run(init)

# 可视化
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion() # plot以后不暂停
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    try:
        ax.lines.remove(lines[0])
    except Exception:
        pass
    sum = sess.run(summaries, feed_dict={xs: x_data, ys: y_data})
    writer.add_summary(sum, global_step=i)
    prediction_value = sess.run(prediction, feed_dict={xs: x_data})
    lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
    plt.pause(0.1)

print("over")