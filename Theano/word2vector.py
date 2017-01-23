
from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utilities import load_data
batchsize = 100


class NextWord(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng = None,
        input = None,
        output = None,
        n_embed = 200,
        n_hidden = 200,
        wW2E = None,
        wE2H = None,
        wH2O = None,
        bH = None,
        bO = None
    ):
        """
        Initialize the Object by specifying the number of visible units (the
        dimension d of the input), the number of hidden units (the dimension
        d' of the latent or hidden space).
        The constructor also receives symbolic variables for the input, weights and
        bias.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for standalone model

        :type n_embed: int
        :param n_embed: number of word-embedding units

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type wW2E: theano.tensor.TensorType
        :param wW2E: Theano variable pointing to a set of weights that should be
                      shared across model architecture;
                      if model is standalone set this to None
        
        :type wE2H: theano.tensor.TensorType
        :param wE2H: Theano variable pointing to a set of weights that should be
                      shared across model architecture;
                      if model is standalone set this to None

        :type bH: theano.tensor.TensorType
        :param bH: Theano variable pointing to a set of biases values (for 
                    visible units) that should be shared across model 
                    architecture; if model is standalone set this to None

        :type bO: theano.tensor.TensorType
        :param bO: Theano variable pointing to a set of biases values (for 
                    visible units) that should be shared across model 
                    architecture; if model is standalone set this to None


        """
        self.numwords = 3 # i.e. we use 3-grams to predict the 4th word
        #batchsize
        self.vocab_size = 250
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_output = self.vocab_size
        
        # used for one-hot-encoding of target vocab index
        self.EYE = theano.tensor.eye(self.vocab_size, dtype=theano.config.floatX)

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            
        if not wW2E: # word_embedding_weights as matrix of size vocab_size X n_embed
            # Initialized with uniform sample
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_wW2E = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.vocab_size + n_embed)),
                    high=4 * numpy.sqrt(6. / (self.vocab_size + n_embed)),
                    size=(self.vocab_size, n_embed)
                ),
                dtype=theano.config.floatX
            )
            wW2E = theano.shared(value=initial_wW2E, name='wW2E', borrow=True)

        if not wE2H: # embed_to_hid_weights as a matrix of size n_embed*numwords X n_hidden
            # Initialized with uniform sample
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_wE2H = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_embed*self.numwords + n_hidden)),
                    high=4 * numpy.sqrt(6. / (n_embed*self.numwords + n_hidden)),
                    size=(n_embed*self.numwords, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            wE2H = theano.shared(value=initial_wE2H, name='wE2H', borrow=True)
            
        if not bH: # hid_bias: Bias of the hidden layer as a matrix of size n_hidden X 1.
            bH = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='bH',
                borrow=True
            )
            
        if not wH2O: # hid_to_output_weights
            # Initialized with uniform sample
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_wH2O = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + self.n_output)),
                    high=4 * numpy.sqrt(6. / (n_hidden + self.n_output)),
                    size=(n_hidden, self.n_output)
                ),
                dtype=theano.config.floatX
            )
            wH2O = theano.shared(value=initial_wH2O, name='wH2O', borrow=True)
        
        if not bO: # output_bias: Bias of the output layer as a matrix of size vocab_size X 1.
            bO = theano.shared(
                value=numpy.zeros(
                    self.n_output,
                    dtype=theano.config.floatX
                ),
                name='bO',
                borrow=True
            )


        self.wW2E = wW2E
        self.wE2H = wE2H
        self.wH2O = wH2O
        # b corresponds to the bias of the hidden activations
        self.bH = bH
        self.bO = bO
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several examples
            self.x = T.dmatrix(name='input') # WARNING : should be Int ?
        else:
            self.x = input
            
        if output is None:
            # we use a vector because we expect a minibatch of several examples
            self.y = T.dvector(name='output') # WARNING : should be Int ? should be Matrix ?
        else:
            self.y = output

        self.params = [self.wW2E, self.wE2H, self.wH2O, self.bH, self.bO]

    def get_embed_values(self, words):
        """ Computes the values of the hidden layer """
        #embedding_layer_state = reshape(...
        #word_embedding_weights(reshape(input_batch, 1, []),:)',...
        #n_embed * numwords, []);
        # see : http://stackoverflow.com/questions/33947726/indexing-tensor-with-index-matrix-in-theano
        
        A = self.wW2E # dim = (vocab, embed)
        #W = words # dim = (batchsize, nbwords)
        
        return self.wW2E[words,:].reshape((-1,self.numwords*self.n_embed))
    
    def get_hidden_values(self, embed):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(embed, self.wE2H) + self.bH)

    def get_output(self, hidden):
        """ Computes the reconstructed input given the values of the hidden layer """
        return T.nnet.softmax(T.dot(hidden, self.wH2O) + self.bO)
    
    def get_cross_entropy(self, output):
        # needs to turn the target (Y) into a one-hot vectors
        Y = self.EYE[self.y,:].reshape((-1,self.vocab_size))
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, CE will be a vector, with one entry per
        #        example in minibatch
        #CE = - T.sum(self.y * T.log(output) + (1 - self.y) * T.log(1 - output), axis=1)
        CE = - T.sum(Y * T.log(output), axis=1)
        # note : CE is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        return CE

    def get_cost_updates(self, learning_rate):
        """ This function computes the cost and the updates for one trainng step of the model """
        embed = self.get_embed_values(self.x)
        hidden = self.get_hidden_values(embed)
        output = self.get_output(hidden)
        CE = self.get_cross_entropy(output)
        
        cost = T.mean(CE)
        
        # compute the gradients of the cost with respect to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ] # WARNING : no momentum at present

        return (cost, updates)


# Temporary snippet to check dimension consistency
def test_snippet_1():
    # Reshape demo based on :
    # http://stackoverflow.com/questions/33947726/indexing-tensor-with-index-matrix-in-theano

    from random import randint

    nbwords = 3
    vocab = 30
    embed = 5
    batchsize = 10

    A = numpy.array([[int(str(i+1)+'000'+str(j+1)) for j in range(embed)] for i in range(vocab)]).reshape(vocab, embed)
    B = numpy.array([randint(0,vocab-1) for i in range(nbwords*batchsize)]).reshape(batchsize, nbwords)

    AA = T.matrix()
    BB = T.imatrix()

    #CC = AA[T.arange(AA.shape[0]).reshape((-1, 1)), T.arange(AA.shape[1]), BB]
    #CC = AA[BB,:]
    #CC = AA[BB,:].reshape((batchsize,nbwords*embed))
    CC = AA[BB,:].reshape((-1,nbwords*embed))

    f = theano.function([AA, BB], CC, allow_input_downcast=True)

    D = f(A.astype(theano.config.floatX), B)

    print(D.shape)
    print(D[0])


def test_NextWord(learning_rate=0.1, training_epochs=5, batch_size=100):

    """
    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    #datasets = load_data(dataset)
    #train_set_x, train_set_y = datasets[0]
    [train_input, train_target, valid_input, valid_target, test_input, test_target, vocab] = load_data(batch_size)

    """ Loads the dataset into shared variables :
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
    """
    borrow = True
    shared_x = theano.shared(numpy.asarray(train_input, dtype=theano.config.floatX),borrow=borrow)
    shared_y = theano.shared(numpy.asarray(train_target, dtype=theano.config.floatX), borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    train_set_x = T.cast(shared_x, 'int32')
    train_set_y = T.cast(shared_y, 'int32')
    
    
    # compute number of minibatches for training, validation and testing
    #n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_train_batches = min(5000, train_input.shape[-1]) # 5 to test

    # allocate symbolic variables for the data
    index = T.lscalar() # index to a [mini]batch, type=INT
    x = T.imatrix('x')
    y = T.imatrix('y')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    nw = NextWord(
        numpy_rng = rng,
        theano_rng = theano_rng,
        input = x,
        output = y,
        n_embed = 50,
        n_hidden = 200
    )
    #print(nw)
    cost, updates = nw.get_cost_updates(learning_rate=learning_rate)
    #print(cost, updates)
    train_nw = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            #x: train_set_x[index * batch_size: (index + 1) * batch_size],
            x: train_set_x[:,:,index],
            #y: train_set_y[index * batch_size: (index + 1) * batch_size]
            y: train_set_y[:,:,index]
        }
    )
    #print(train_nw)
    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in range(training_epochs):
        # go through training set
        c = []
        for batch_index in range(n_train_batches):
            #print(batch_index)
            c.append(train_nw(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c, dtype='float64'))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print('The code ran for %.2fm' % ((training_time) / 60.))
    
