
from __future__ import print_function
import numpy as np
import tensorflow as tf


batch_size = 100
kappaNN = 0.001*2
kappaLGSTC = 0.001*5

numwords = 3
vocab_size = 250
n_embed = 150
n_hidden = 100
n_output = vocab_size


graph = tf.Graph()
with graph.as_default():
    
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.int32, shape=(numwords, batch_size))
    tf_train_labels = tf.placeholder(tf.int32, shape=(1, batch_size))
    #tf_valid_dataset = tf.constant(valid_dataset)
    #tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weightsW2E = tf.Variable(tf.truncated_normal([vocab_size, n_embed]))
    weightsE2H = tf.Variable(tf.truncated_normal([n_embed*numwords, n_hidden]))
    biasesH = tf.Variable(tf.zeros([n_hidden]))
    weightsH2O = tf.Variable(tf.truncated_normal([n_hidden, n_output]))
    biasesO = tf.Variable(tf.zeros([n_output]))

    # Training computation.
    train_embed = tf.reshape(tf.gather(weightsW2E, tf_train_dataset, validate_indices=None, name=None),
                             [-1,numwords*n_embed])
    train_logitsH = tf.matmul(train_embed, weightsE2H) + biasesH
    train_H = tf.sigmoid(train_logitsH)
    train_logitsO = tf.matmul(train_H, weightsH2O) + biasesO
    train_predictionO = tf.nn.softmax(train_logitsO) # Info : not used in cross-entropy function
    print(train_logitsO)
    
    train_target = tf.reshape(tf.gather(tf.eye(vocab_size), tf_train_labels, validate_indices=None, name=None), [batch_size,-1])
    print(train_target)
        
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logitsO, tf_train_labels))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logitsO, train_target))
    param1_l2 = tf.nn.l2_loss(weightsE2H)# + tf.nn.l2_loss(biasesH)
    param2_l2 = tf.nn.l2_loss(weightsH2O)# + tf.nn.l2_loss(biasesO)
    reg_loss = loss + kappaNN * (param1_l2 + param2_l2)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(reg_loss)

    '''
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
    test_predictionO = tf.nn.softmax(test_logitsO)'''



num_steps = 1501 # to debug

from utilities import load_data

[train_input, train_target, valid_input, valid_target, test_input, test_target, vocab] = load_data(batch_size)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Generate a minibatch.
        batch_data = train_input[:,:,step].astype(int)
        batch_labels = train_target[:,:,step].astype(int)
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_predictionO], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            #print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))