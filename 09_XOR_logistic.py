# Implement code for Sung Kim's TF lecture. See https://www.youtube.com/watch?v=9i7FBbcZPMA&feature=youtu.be

import numpy as np
import tensorflow as tf

xy = np.loadtxt('09_train.txt', unpack=True)

# Need to change data structure. THESE LINES ARE DIFFERNT FROM Video BUT IT MAKES THIS CODE WORKS!
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform( [1, len(x_data)], -1.0, 1.0))

# Hypotheses 
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))

# Cost function 
cost = -tf.reduce_mean( Y*tf.log(hypothesis) + (1-Y)*tf.log(1.-hypothesis))

# Minimize cost.
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Initializa all variables.
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for step in range(8001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 1000 == 0:
            print(
                    step,\
                    sess.run(cost, feed_dict={X:x_data, Y:y_data}),\
                    sess.run(W)
                    )
    # Test model
    correct_prediction = tf.equal( tf.floor(hypothesis+0.5), Y)
    accuracy = tf.reduce_mean(tf.cast( correct_prediction, "float" ) )

    # Check accuracy
    print( sess.run( [hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy],\
                        feed_dict={X:x_data, Y:y_data}) )
    print( "Accuracy:", accuracy.eval({X:x_data, Y:y_data}) )
