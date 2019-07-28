import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 超参
lr = 0.001 # 学习率
training_iters = 100000
batch_size = 128
# display_step = 10

n_inputs = 28 # 图片28×28
n_steps = 28  # 时间步，28步
n_hidden_units = 128 # 隐藏层神经元数目
n_classes = 10 # 分类0～9

#
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 定义权重
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
     'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, weights, biases):
    #
    # X: (128 batches, 28 steps, 28 inputs) => (128*28, 28 inputs),三维转二维
    # 将图片展开成一维序列（长为128×28, 共28个输入序列）
    X = tf.reshape(X, [-1, n_inputs])

    # X_in => (128 batches * 28 steps, 128 hidden_units)
    X_in = tf.matmul(X, weights['in']) + biases['in']

    # X_in => (128 batches, 28 steps, 128 hidden_units)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    # forget_bias为1,表示不希望forget前面的东西；state_is_tuple为True，即生成一个元组：（主线剧情，分线剧情）
    # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    # lstm细胞分为主线state和分线state; 初始化为全零
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # 输出层
    results = tf.matmul(states[1], weights['out']) + biases['out']
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
        step += 1