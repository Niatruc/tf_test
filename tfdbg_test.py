import tensorflow as tf
from tensorflow.python import debug as tfdbg

v1 = tf.Variable([1,2,3], name='a')
v2 = tf.Variable([4,5,6], name='b')
v1_plus_v2 = v1 + v2

sess = tf.Session()
sess = tfdbg.TensorBoardDebugWrapperSession(sess, 'localhost:6064')

init = tf.initialize_all_variables()
sess.run(init)
sess.run(v1_plus_v2)

