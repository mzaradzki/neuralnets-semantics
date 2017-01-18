
import math
import numpy as np
from numpy import size, reshape
from numpy.matlib import repmat



def logistic(inputs_to_logistic):
    return 1. / (1. + np.exp(-inputs_to_logistic));



def softmax(inputs_to_softmax):
    nbrows = inputs_to_softmax.shape[0];
    # Subtract maximum (does not change result) to prevent overflow when computing exponentials
    inputs_to_softmax = inputs_to_softmax - repmat(np.max(inputs_to_softmax, 0), nbrows, 1);
    # Compute exponentials
    output_layer_state = np.exp(inputs_to_softmax);
    # Normalize to get probability distribution.
    return output_layer_state / repmat(np.sum(output_layer_state, 0), nbrows, 1);



def fprop(input_batch, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias):
    # This method forward propagates through a neural network.
    # Inputs:
    #   input_batch: The input data as a matrix of size numwords X batchsize where,
    #     numwords is the number of words, batchsize is the number of data points.
    #     So, if input_batch(i, j) = k then the ith word in data point j is word
    #     index k of the vocabulary.
    #
    #   word_embedding_weights: Word embedding as a matrix of size
    #     vocab_size X numhid1, where vocab_size is the size of the vocabulary
    #     numhid1 is the dimensionality of the embedding space.
    #
    #   embed_to_hid_weights: Weights between the word embedding layer and hidden
    #     layer as a matrix of soze numhid1*numwords X numhid2, numhid2 is the
    #     number of hidden units.
    #
    #   hid_to_output_weights: Weights between the hidden layer and output softmax
    #               unit as a matrix of size numhid2 X vocab_size
    #
    #   hid_bias: Bias of the hidden layer as a matrix of size numhid2 X 1.
    #
    #   output_bias: Bias of the output layer as a matrix of size vocab_size X 1.
    #
    # Outputs:
    #   embedding_layer_state: State of units in the embedding layer as a matrix of
    #     size numhid1*numwords X batchsize
    #
    #   hidden_layer_state: State of units in the hidden layer as a matrix of size
    #     numhid2 X batchsize
    #
    #   output_layer_state: State of units in the output layer as a matrix of size
    #     vocab_size X batchsize
    #

    [numwords, batchsize] = input_batch.shape;
    [vocab_size, numhid1] = word_embedding_weights.shape;
    numhid2 = size(embed_to_hid_weights, 1);

    # COMPUTE STATE OF WORD EMBEDDING LAYER.
    # Look up the inputs word indices in the word_embedding_weights matrix.
    #reshape(input_batch, (1,-1)); # DEBUG LINE
    #word_embedding_weights[reshape(input_batch, (1,-1)),:]; # DEBUG LINE
    embedding_layer_state = reshape(word_embedding_weights[reshape(input_batch, (1,-1)),:].T, (numhid1 * numwords, -1));

    # COMPUTE STATE OF HIDDEN LAYER.
    # a) Compute inputs to hidden units.
    inputs_to_hidden_units = np.dot(embed_to_hid_weights.T , embedding_layer_state) + repmat(hid_bias, 1, batchsize);
    # b) Apply logistic activation function.
    hidden_layer_state = logistic(inputs_to_hidden_units); # dim=(numhid2, batchsize)

    # COMPUTE STATE OF OUTPUT LAYER.
    # a) Compute inputs to softmax.
    inputs_to_softmax = np.dot(hid_to_output_weights.T, hidden_layer_state) + repmat(output_bias, 1, batchsize); # dim=(vocab_size, batchsize)
    # b) Apply softmax function
    output_layer_state = softmax(inputs_to_softmax);

    return [embedding_layer_state, hidden_layer_state, output_layer_state];



def predict_next_word(word1, word2, word3, model, k):
    # Predicts the next word.
    # Inputs:
    #   word1: The first word as a string.
    #   word2: The second word as a string.
    #   word3: The third word as a string.
    #   model: Model returned by the training script.
    #   k: The k most probable predictions are shown.
    # Example usage:
    #   predict_next_word('john', 'might', 'be', model, 3);
    #   predict_next_word('life', 'in', 'new', model, 3);

    try:
        id1 = model['vocab'].flatten().tolist().index(word1);
    except:
        raise ValueError("Word %s not in vocabulary." % word1);

    try:
        id2 = model['vocab'].flatten().tolist().index(word2);
    except:
        raise ValueError("Word %s not in vocabulary." % word2);

    try:
        id3 = model['vocab'].flatten().tolist().index(word3);
    except:
        raise ValueError("Word %s not in vocabulary." % word3);

    triplet = np.array([id1, id2, id3]); # WARNING : we transpose ?
    [embedding_layer_state, hidden_layer_state, output_layer_state] = fprop(triplet, model.word_embedding_weights, model.embed_to_hid_weights, model.hid_to_output_weights, model.hid_bias, model.output_bias);

    raise ValueError('Function code is not complete');
    [prob, indices] = sort(output_layer_state, 'descend');

    for i in range(k):
        print "%s %s %s %s Prob: %.5f", word1, word2, word3, model['vocab'][indices[i]], prob[i];



def display_nearest_words(word, model, k):
    # Shows the k-nearest words to the query word.
    # Inputs:
    #   word: The query word as a string.
    #   model: Model returned by the training script.
    #   k: The number of nearest words to display.
    # Example usage:
    #   display_nearest_words('school', model, 10);

    word_embedding_weights = model['word_embedding_weights'];
    
    try:
        wid = model['vocab'].flatten().tolist().index(word);
    except:
        raise ValueError("Word %s not in vocabulary." % word);

    # Compute distance to every other word.
    vocab_size = size(model['vocab'], 1);
    word_rep = word_embedding_weights[wid, :];
    diff = word_embedding_weights - repmat(word_rep, vocab_size, 1);
    distance = math.sqrt(np.sum(diff * diff, 1));

    # Sort by distance.
    raise ValueError('Function code is not complete');
    [d, order] = sort(distance);
    order = order[1:k+1];  # The nearest word is the query word itself, skip that.
    
    for i in range(k):
        print "%s %.2f" % (model['vocab'][order[i]], distance[order[i]]);

