import tensorflow as tf
import numpy as np
import input_data
import os

def init_weights(shape):
    w = tf.Variable(tf.random_normal(shape, stddev=0.01))
    return w

def model(X, w1, w2, w3, w4, w_out, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1,1,1,1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1,1,1,1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1,1,1,1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    pyx = tf.matmul(l4, w_out)
    return pyx

batch_size = 128
test_size = 256

training_epoch = 15
display_step = 10
#batch_size = 100

# MNIST variable initialize
mnist = input_data.read_data_sets("MNIST", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

print len(trX)

# Input, Output
X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

# weights [row, col, depth, filter #]
w1 = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128*4*4, 625])
w_out = init_weights([625, 10])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
"""
l1a = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1,1,1,1], padding='SAME'))
l1 = tf.nn.max_pool(l1a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
l1 = tf.nn.dropout(l1, p_keep_conv)

l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1,1,1,1], padding='SAME'))
l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
l2 = tf.nn.dropout(l2, p_keep_conv)

l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1,1,1,1], padding='SAME'))
l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
l3 = tf.nn.dropout(l3, p_keep_conv)

l4 = tf.nn.relu(tf.matmul(l3, w4))
l4 = tf.nn.dropout(l4, p_keep_hidden)
py_x = tf.matmul(l4, w_out)
"""

py_x = model(X, w1, w2, w3, w4, w_out, p_keep_conv, p_keep_hidden)

#print py_x

# using Softmax & optimization
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
optimizer = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

#accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1)), tf.float32))

ckpt_dir = "./save"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

global_step = tf.Variable(0, name='global_step', trainable=False)

# after declare all tf.Variables, call saver
saver = tf.train.Saver()

with tf.Session() as sess:

    tf.initialize_all_variables().run()

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    start = global_step.eval()
    print("Start from:", start)

    for i in range(start, 100):
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(optimizer, feed_dict={X:trX[start:end], Y:trY[start:end], p_keep_conv:0.8, p_keep_hidden:0.5})

            if start%512==0:
                print '[start:', "%6d"%(start),
                print ', cost:', sess.run(cost, feed_dict={X:trX[start:end], Y:trY[start:end], p_keep_conv:0.8, p_keep_hidden:0.5}),
                print ']'

        global_step.assign(i).eval()    # set and eval(update) global_step with index(i)
        saver.save(sess, ckpt_dir+"/model.ckpt", global_step=global_step)
        print(i, np.mean(np.argmax(teY, axis=1) ==\
                            sess.run(predict_op, feed_dict={X: teX,\
                                                            p_keep_conv:1.0, p_keep_hidden:1.0})))

"""
        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print 'error rate:',
        print i, np.mean(np.argmax(teY[test_indices], axis=1) ==\
                        sess.run(predict_op, feed_dict={X: teX[test_indices],\
                        p_keep_conv:1.0, p_keep_hidden:1.0}))
"""

"""
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        sess.run(optimizer, feed_dict={X:batch_xs, Y:batch_ys, p_keep_conv:0.8, p_keep_hidden:0.5})

    print 'Accuracy:', accuracy.eval({X:mnist.test.images, Y:mnist.test.labels, p_keep_conv:1.0, p_keep_hidden:1.0})

"""








