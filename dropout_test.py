import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 载入数据
digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)


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
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob) #
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


keep_prob = tf.placeholder(tf.float32) # 要保持多少不被丢掉的百分比

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 64], name='x_input') #tf.float32一定要加上
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input') #0123456789

l1 = add_layer(xs, 64, 100, 1, activation_function=tf.nn.tanh)
prediction = add_layer(l1, 100, 10, 2, activation_function=tf.nn.softmax)

# 用交叉熵函数计算损失
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-10,50.0)), reduction_indices=[1]))
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    # tf.summary.histogram()
    tf.summary.scalar('loss', loss)

# 梯度下降
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.6).minimize(loss) # 0.1是学习效率(learning rate)

# 初始化
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 日志记录器
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)
summaries = tf.summary.merge_all()

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})

    if i % 50 == 0:
        # prediction_value = sess.run(prediction, feed_dict={xs: x})
        train_result = sess.run(summaries, feed_dict={xs: x_train, ys: y_train, keep_prob: 1}) # 记录的时候步drop掉任何东西
        test_result = sess.run(summaries, feed_dict={xs: x_test, ys: y_test, keep_prob: 1}) # 记录的时候步drop掉任何东西
        train_writer.add_summary(train_result, global_step=i)
        test_writer.add_summary(test_result, global_step=i)


print("over")