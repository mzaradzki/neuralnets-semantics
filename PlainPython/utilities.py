

import math
import numpy as np
from numpy import size, reshape
from scipy.io import loadmat


# DISCLAIMER : Translated from G. Hinton Octave code
def load_data(N):
    # This method loads the training, validation and test set.
    # It also divides the training set into mini-batches.
    # Inputs:
    #   N: Mini-batch size.
    # Outputs:
    #   train_input: An array of size D X N X M, where
    #                 D: number of input dimensions (in this case, 3).
    #                 N: size of each mini-batch (in this case, 100).
    #                 M: number of minibatches.
    #   train_target: An array of size 1 X N X M.
    #   valid_input: An array of size D X number of points in the validation set.
    #   test: An array of size D X number of points in the test set.
    #   vocab: Vocabulary containing index to word mapping.
    
    data = loadmat('data.mat', matlab_compatible=False, struct_as_record=False)['data'][0,0]; # WARNING : special utility for .mat Matlab files
    numdims = size(data.trainData, 0);
    D = numdims - 1;
    M = math.floor(size(data.trainData, 1) / N);
    train_input = reshape(data.trainData[0:D, 0:N * M], (D, N, M));
    train_target = reshape(data.trainData[D, 0:N * M], (1, N, M));
    valid_input = data.validData[0:D, :];
    valid_target = data.validData[D, :];
    test_input = data.testData[0:D, :];
    test_target = data.testData[D, :];
    vocab = data.vocab;
    if True:
        for i in range(vocab.shape[1]):
            vocab[0,i] = vocab[0,i][0];
    
    return [train_input, train_target,
            valid_input, valid_target,
            test_input, test_target,
            vocab];



# Translated from G. Hinton Octave code
def word_distance(word1, word2, model):
    # Shows the L2 distance between word1 and word2 in the word_embedding_weights.
    # Inputs:
    #   word1: The first word as a string.
    #   word2: The second word as a string.
    #   model: Model returned by the training script.
    # Example usage:
    #   word_distance('school', 'university', model);

    word_embedding_weights = model['word_embedding_weights'];
    vocab = model['vocab'];
    
    try:
        id1 = vocab.flatten().tolist().index(word1);
    except:
        raise ValueError("Word %s not in vocabulary." % word1);

    try:
        id2 = vocab.flatten().tolist().index(word2);
    except:
        raise ValueError("Word %s not in vocabulary." % word2);

    word_rep1 = word_embedding_weights[id1, :];
    word_rep2 = word_embedding_weights[id2, :];
    diff = word_rep1 - word_rep2;
    return math.sqrt(np.sum(diff * diff));
