

from __future__ import print_function
import numpy as np
import tensorflow as tf


batch_size = 128
num_relus = 1024
kappaNN = 0.001*2
kappaLGSTC = 0.001*5

graph = tf.Graph()
with graph.as_default():

	# Input data. For the training data, we use a placeholder that will be fed
	# at run time with a training minibatch.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	# Variables.
	weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_relus]))
	biases1 = tf.Variable(tf.zeros([num_relus]))
	weights2 = tf.Variable(tf.truncated_normal([num_relus, num_labels]))
	biases2 = tf.Variable(tf.zeros([num_labels]))

	# Training computation.
	train_logits1 = tf.matmul(tf_train_dataset, weights1) + biases1
	train_x2 = tf.nn.relu(train_logits1) # non-linear hidden layer
	train_logits2 = tf.matmul(train_x2, weights2) + biases2

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits2, tf_train_labels))
	param1_l2 = tf.nn.l2_loss(weights1)# + tf.nn.l2_loss(biases1)
	param2_l2 = tf.nn.l2_loss(weights2)# + tf.nn.l2_loss(biases2)
	reg_loss = loss + kappaNN * (param1_l2 + param2_l2)

	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(reg_loss)

	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(train_logits2)
	#
	valid_logits1 = tf.matmul(tf_valid_dataset, weights1) + biases1
	valid_x2 = tf.nn.relu(valid_logits1)
	valid_logits2 = tf.matmul(valid_x2, weights2) + biases2
	valid_prediction = tf.nn.softmax(valid_logits2)
	#
	test_logits1 = tf.matmul(tf_test_dataset, weights1) + biases1
	test_x2 = tf.nn.relu(test_logits1)
	test_logits2 = tf.matmul(test_x2, weights2) + biases2
	test_prediction = tf.nn.softmax(test_logits2)

num_steps = 3001

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print("Initialized")
	for step in range(num_steps):
	    # Pick an offset within the training data, which has been randomized.
	    # Note: we could use better randomization across epochs.
	    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
	    #offset = step % 2
	    # Generate a minibatch.
	    batch_data = train_dataset[offset:(offset + batch_size), :]
	    batch_labels = train_labels[offset:(offset + batch_size), :]
	    # Prepare a dictionary telling the session where to feed the minibatch.
	    # The key of the dictionary is the placeholder node of the graph to be fed,
	    # and the value is the numpy array to feed to it.
	    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
	    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
	    if (step % 500 == 0):
	        print("Minibatch loss at step %d: %f" % (step, l))
	        print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
	        print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
	print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))