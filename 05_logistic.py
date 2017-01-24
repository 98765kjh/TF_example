import tensorflow as tf
import numpy as np

xy = np.loadtxt('05_train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, 1.0))

#hypothesis
h = tf.matmul(W,X)
hypo = tf.div(1., 1.+tf.exp(-h))

#cost func
cost = -tf.reduce_mean(Y*tf.log(hypo) + (1-Y)*tf.log(1-hypo))

#minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(50001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step%1000 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

print '-----------------------------------------'
# test
print sess.run(hypo, feed_dict={X:[[1],[2],[2]]})>0.5
print sess.run(hypo, feed_dict={X:[[1],[5],[5]]})>0.5

print sess.run(hypo, feed_dict={X:[[1, 1],[4, 3],[3, 5]]})>0.5
