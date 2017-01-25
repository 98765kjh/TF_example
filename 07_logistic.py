import tensorflow as tf
import input_data

X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([28*28, 10]))
b = tf.Variable(tf.zeros([10]))

hypo = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(tf.reduce_sum(-Y*tf.log(hypo), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
init = tf.initialize_all_variables()

training_epoch = 25
display_step = 1
batch_size = 100
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

with tf.Session() as session:
    session.run(init)

    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            session.run(optimizer, feed_dict = {X:batch_xs, Y:batch_ys})
            # Compute average loss
            avg_cost += session.run(cost, feed_dict = {X:batch_xs, Y:batch_ys}) / total_batch

        # Display logs per epoch steps
        if epoch % display_step == 0:
            print "Epoch:", '%04d' %(epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypo, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({X:mnist.test.images, Y:mnist.test.labels}))
