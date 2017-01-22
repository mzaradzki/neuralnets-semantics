

from __future__ import print_function
import numpy as np
import tensorflow as tf


batch_size = 128
num_relus = 1024
kappaNN = 0.001*2
kappaLGSTC = 0.001*5

graph = tf.Graph()
with graph.as_default():
    
    #self.n_embed
    #self.numwords
    #self.n_hidden
    #self.n_output
    
    #weights1 = weightsE2H => alias
    #biases1 = biasesE2H = biasesH 
    #weights2 = weightsH2O => alias
    #biases2 = biasesH2O = biasesO => alias
    #train_logits1 = train_logitsH => alias
    #train_x2 = train_H => alias
    #train_logits2 = train_logitsO => alias

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weightsE2H = tf.Variable(tf.truncated_normal([self.n_embed*self.numwords, self.n_hidden]))
    biasesH = tf.Variable(tf.zeros([self.n_hidden]))
    weightsH2O = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_output]))
    biasesO = tf.Variable(tf.zeros([self.n_output]))

    # Training computation.
    train_logitsH = tf.matmul(tf_train_dataset, weightsE2H) + biasesH
    train_H = tf.nn.relu(train_logitsH) # non-linear hidden layer
    train_logitsO = tf.matmul(train_H, weightsH2O) + biasesO
    #train_predictionO = tf.nn.softmax(train_logitsO) # Reference only : not used in cross-entropy function

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logitsO, tf_train_labels))
    param1_l2 = tf.nn.l2_loss(weightsE2H)# + tf.nn.l2_loss(biasesH)
    param2_l2 = tf.nn.l2_loss(weightsH2O)# + tf.nn.l2_loss(biasesO)
    reg_loss = loss + kappaNN * (param1_l2 + param2_l2)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(reg_loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(train_logitsO)
    #
    valid_logitsH = tf.matmul(tf_valid_dataset, weightsE2H) + biasesE2H
    valid_H = tf.nn.relu(valid_logitsH)
    valid_logitsO = tf.matmul(valid_H, weightsH2O) + biasesH2O
    valid_predictionO = tf.nn.softmax(valid_logitsO)
    #
    test_logitsH = tf.matmul(tf_test_dataset, weightsE2H) + biasesE2H
    test_H = tf.nn.relu(test_logitsH)
    test_logitsO = tf.matmul(test_H, weightsH2O) + biasesH2O
    test_predictionO = tf.nn.softmax(test_logitsO)


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