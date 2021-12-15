import tensorflow as tf

a = tf.constant([[1, 2, 3]])
b = tf.transpose(a)
c = a + b

with tf.Session() as sess:
    print(sess.run(c))