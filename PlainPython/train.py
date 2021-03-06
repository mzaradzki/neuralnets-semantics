
import math
import time
import numpy as np
from numpy.random import randn
from numpy import size

from neuralnet import fprop
from utilities import load_data

def fflush(x):
    return None; # TBD based on Octave documentation

# This function trains a neural network language model.
def train(epochs):
    # Inputs:
    #   epochs: Number of epochs to run.
    # Output:
    #   model: A struct containing the learned weights and biases and vocabulary.

    start_time = time.time();
   

    # SET HYPERPARAMETERS HERE.
    batchsize = 100;  # Mini-batch size.
    learning_rate = 0.1;  # Learning rate; default = 0.1.
    momentum = 0.9;  # Momentum; default = 0.9.
    numhid1 = 50;  # Dimensionality of embedding space; default = 50.
    numhid2 = 200;  # Number of units in hidden layer; default = 200.
    init_wt = 0.01;  # Standard deviation of the normal distribution
                    # which is sampled to get the initial weights; default = 0.01

    # VARIABLES FOR TRACKING TRAINING PROGRESS.
    show_training_CE_after = 100;
    show_validation_CE_after = 1000;

    # LOAD DATA.
    [train_input, train_target, valid_input, valid_target, test_input, test_target, vocab] = load_data(batchsize);
    [numwords, batchsize, numbatches] = train_input.shape;
    vocab_size = size(vocab, 1);

    # INITIALIZE WEIGHTS AND BIASES.
    word_embedding_weights = init_wt * randn(vocab_size, numhid1);
    embed_to_hid_weights = init_wt * randn(numwords * numhid1, numhid2);
    hid_to_output_weights = init_wt * randn(numhid2, vocab_size);
    hid_bias = np.zeros((numhid2, 1));
    output_bias = np.zeros((vocab_size, 1));

    word_embedding_weights_delta = np.zeros((vocab_size, numhid1));
    #word_embedding_weights_gradient = np.zeros((vocab_size, numhid1));
    embed_to_hid_weights_delta = np.zeros((numwords * numhid1, numhid2));
    hid_to_output_weights_delta = np.zeros((numhid2, vocab_size));
    hid_bias_delta = np.zeros((numhid2, 1));
    output_bias_delta = np.zeros((vocab_size, 1));
    expansion_matrix = np.eye(vocab_size);
    counter = 0;
    tiny = math.exp(-30);


    def ModelCrossEntropy(_input, _target):
        assert _input.shape[0] == numwords;
        assert _target.shape == (_input.shape[1],);
        assert word_embedding_weights.shape == (vocab_size, numhid1);
        assert embed_to_hid_weights.shape == (numwords * numhid1, numhid2);
        assert hid_to_output_weights.shape == (numhid2, vocab_size);
        assert hid_bias.shape == (numhid2, 1);
        assert output_bias.shape == (vocab_size, 1);
        [embedding_layer_state, hidden_layer_state, output_layer_state] = fprop(_input, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias);
        datasetsize = size(_input, 1);
        expanded_valid_target = expansion_matrix[:, _target];
        #print expanded_valid_target.shape;
        #print np.log(output_layer_state + tiny).shape;
        return -np.sum(expanded_valid_target * np.log(output_layer_state + tiny)) / datasetsize;

    def fprop_gradient(_input, _target, _counter, _this_chunk_CE, _trainset_CE):
        # FORWARD PROPAGATE.
        # Compute the state of each layer in the network given the input batch
        # and all weights and biases
        #print word_embedding_weights.shape;
        [embedding_layer_state, hidden_layer_state, output_layer_state] = fprop(_input, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias);
        #print output_layer_state.shape;
        # COMPUTE DERIVATIVE.
        # Expand the target to a sparse 1-of-K vector.
        expanded_target_batch = expansion_matrix[:, _target][:,0,:]; # WARNING : strange indices
        #print expanded_target_batch.shape;
        #print type(expanded_target_batch);
        #print type(output_layer_state);
        # Compute derivative of cross-entropy loss function.
        error_deriv = output_layer_state - expanded_target_batch;
        #print error_deriv.shape;
        #print type(error_deriv.shape);

        # MEASURE LOSS FUNCTION.
        CE = -np.sum(np.sum(expanded_target_batch * np.log(output_layer_state + tiny))) / batchsize;
        
        _counter = _counter + 1;
        _this_chunk_CE = _this_chunk_CE + (CE - _this_chunk_CE) / _counter;
        _trainset_CE = _trainset_CE + (CE - _trainset_CE) / (m+1);
        if (m % show_training_CE_after) == 0:
            print "Batch %d Train CE %.3f \n" % (m, _this_chunk_CE);
            _counter = 0;
            _this_chunk_CE = 0;

        if True:
            fflush(1);


        # BACK PROPAGATE.
        # OUTPUT LAYER.
        hid_to_output_weights_gradient =  np.dot(hidden_layer_state, error_deriv.T);
        assert hid_to_output_weights_gradient.shape == hid_to_output_weights_delta.shape;
        output_bias_gradient = np.sum(error_deriv, 1).reshape((-1,1));
        #print output_bias_gradient.shape, output_bias_delta.shape;
        assert output_bias_gradient.shape == output_bias_delta.shape;
        back_propagated_deriv_1 = np.dot(hid_to_output_weights, error_deriv) * hidden_layer_state * (1 - hidden_layer_state);

        # HIDDEN LAYER.
        embed_to_hid_weights_gradient = np.dot(embedding_layer_state, back_propagated_deriv_1.T); # dim=(numhid1 * numwords, numhid2)
        assert embed_to_hid_weights_gradient.shape == embed_to_hid_weights_delta.shape;
        hid_bias_gradient = np.sum(back_propagated_deriv_1, 1).reshape((-1,1)); # dim=(numhid2, 1)
        #print hid_bias_gradient.shape, hid_bias_delta.shape;
        assert hid_bias_gradient.shape == hid_bias_delta.shape;
        
        back_propagated_deriv_2 = np.dot(embed_to_hid_weights, back_propagated_deriv_1); # dim=(numhid2, batchsize)

        #word_embedding_weights_gradient[:] = 0;
        word_embedding_weights_gradient = np.zeros((vocab_size, numhid1));
        #print word_embedding_weights_gradient.shape;
        # EMBEDDING LAYER.
        for w in range(numwords):
            #print expansion_matrix[:, input_batch[w, :]].shape;
            #print back_propagated_deriv_2[w*numhid1 : (w+1)*numhid1, :].T.shape;
            word_embedding_weights_gradient = word_embedding_weights_gradient + np.dot(expansion_matrix[:, input_batch[w, :]], back_propagated_deriv_2[w*numhid1 : (w+1)*numhid1, :].T);
        assert word_embedding_weights_gradient.shape == word_embedding_weights_delta.shape;
        
        return [hid_to_output_weights_gradient, output_bias_gradient, embed_to_hid_weights_gradient, hid_bias_gradient, word_embedding_weights_gradient];


    # TRAIN.
    for epoch in range(epochs):
        print "Epoch %d" % epoch;
        this_chunk_CE = 0;
        trainset_CE = 0;
        # LOOP OVER MINI-BATCHES.
        for m in range(numbatches):
            input_batch = train_input[:, :, m];
            target_batch = train_target[:, :, m];
    
            [hid_to_output_weights_gradient, output_bias_gradient, embed_to_hid_weights_gradient, hid_bias_gradient, word_embedding_weights_gradient] = fprop_gradient(input_batch, target_batch, counter, this_chunk_CE, trainset_CE);
            
            # UPDATE DELTA WITH GRADIENT AND MOMENTUM METHOD
            word_embedding_weights_delta = momentum * word_embedding_weights_delta + word_embedding_weights_gradient / batchsize;
            embed_to_hid_weights_delta = momentum * embed_to_hid_weights_delta + embed_to_hid_weights_gradient / batchsize;
            hid_to_output_weights_delta = momentum * hid_to_output_weights_delta + hid_to_output_weights_gradient / batchsize;
            hid_bias_delta = momentum * hid_bias_delta + hid_bias_gradient / batchsize;
            output_bias_delta = momentum * output_bias_delta + output_bias_gradient / batchsize;

            # UPDATE PARAMETERS WITH DELTA AND GRADIENT-DESCENT METHOD
            word_embedding_weights = word_embedding_weights - learning_rate * word_embedding_weights_delta;
            embed_to_hid_weights = embed_to_hid_weights - learning_rate * embed_to_hid_weights_delta;
            hid_to_output_weights = hid_to_output_weights - learning_rate * hid_to_output_weights_delta;
            hid_bias = hid_bias - learning_rate * hid_bias_delta;
            output_bias = output_bias - learning_rate * output_bias_delta;
    
            # VALIDATE.
            if ((m != 0) and ((m % show_validation_CE_after) == 0)): # skip 0 to avoid / by 0
                print "Running validation ...";
                if True:
                    fflush(1);
                CE = ModelCrossEntropy(valid_input, valid_target);
                #print " Validation CE %.3f" % CE;
                if True:
                    fflush(1);
             # end of validation
        # end of batch
        print "Average Training CE %.3f" % trainset_CE;
    # end of training
    print "Finished Training.";
    if True:
        fflush(1);

    print "Final Training CE %.3f" % trainset_CE;

    # EVALUATE ON VALIDATION SET.
    print "Running validation ...";
    if True:
        fflush(1);


    CE = ModelCrossEntropy(valid_input, valid_target);
    print "Final Validation CE %.3f" % CE;
    if True:
        fflush(1);


    # EVALUATE ON TEST SET.
    print "Running test ...";
    if True:
        fflush(1);

    CE = ModelCrossEntropy(test_input, test_target);
    print "Final Test CE %.3f" % CE;
    if True:
        fflush(1);


    model = {};
    model['word_embedding_weights'] = word_embedding_weights;
    model['embed_to_hid_weights'] = embed_to_hid_weights;
    model['hid_to_output_weights'] = hid_to_output_weights;
    model['hid_bias'] = hid_bias;
    model['output_bias'] = output_bias;
    model['vocab'] = vocab;

    end_time = time.time();
    diff = end_time - start_time;

    print "Training took %.2f seconds" % diff;

    return model;
