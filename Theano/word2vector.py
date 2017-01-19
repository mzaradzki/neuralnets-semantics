

from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image


class dA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        #numwords
        #batchsize
        #vocab_size
        #numhid1
        #numhid2 # numhid2 is the number of hidden units.
        self.n_embed = n_embed
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            
        #word_embedding_weights: Word embedding as a matrix of size
        #vocab_size X numhid1, where vocab_size is the size of the vocabulary
        #numhid1 is the dimensionality of the embedding space.
        
        if not wW2E: # word_embedding_weights as matrix of size vocab_size X numhid1
            # Initialized with uniform sample
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_sW2E = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (vocab_size + numhid1)),
                    high=4 * numpy.sqrt(6. / (vocab_size + numhid1)),
                    size=(vocab_size, numhid1)
                ),
                dtype=theano.config.floatX
            )
            wW2E = theano.shared(value=initial_wE2H, name='wW2E', borrow=True)

        if not wE2H: # embed_to_hid_weights as a matrix of size numhid1*numwords X numhid2
            # Initialized with uniform sample
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_wE2H = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_embed + n_hidden)),
                    high=4 * numpy.sqrt(6. / (n_embed + n_hidden)),
                    size=(n_embed, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            wE2H = theano.shared(value=initial_wE2H, name='wE2H', borrow=True)
            
        if not wH2O: # hid_to_output_weights
            # Initialized with uniform sample
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_wH2O = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_output)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_output)),
                    size=(n_hidden, n_output)
                ),
                dtype=theano.config.floatX
            )
            wH2O = theano.shared(value=initial_wH2O, name='wH2O', borrow=True)
        
        if not bH: # hid_bias: Bias of the hidden layer as a matrix of size numhid2 X 1.
            bH = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='bH',
                borrow=True
            )
        
        if not bO: # output_bias: Bias of the output layer as a matrix of size vocab_size X 1.
            bO = theano.shared(
                value=numpy.zeros(
                    n_output,
                    dtype=theano.config.floatX
                ),
                name='bO',
                borrow=True
            )


        self.wW2E = wW2E
        self.wE2H = wE2H
        self.wH2O = wH2O
        # b corresponds to the bias of the hidden
        self.bH = bH
        self.bO = bO
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.wE2H, self.wH2O, self.bH, self.bO]

    def get_embed_values(self, words):
        """ Computes the values of the hidden layer """
        #embedding_layer_state = reshape(...
        #word_embedding_weights(reshape(input_batch, 1, []),:)',...
        #numhid1 * numwords, []);
    
    def get_hidden_values(self, embed):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(embed, self.wE2H) + self.bH)

    def get_output(self, hidden):
        """ Computes the reconstructed input given the values of the hidden layer """
        #return T.nnet.sigmoid(T.dot(hidden, self.wH2O) + self.bO)
        raise ValueError('need to return softmax');

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng step of the dA """

        embed = self.get_embed_values(self.x)
        hidden = self.get_hidden_values(embed)
        ouput = self.get_output(hidden)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)


def test_dA(learning_rate=0.1, training_epochs=15,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='dA_plots'):

    """
    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    # end-snippet-2

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c, dtype='float64'))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print(('The no corruption code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)
    
