'''
    LLNet Color Implementation
    Written by Kin Gwn Lore
    Libraries needed for llnet_color.py
    2/25/2019
'''

import numpy
import theano
import theano.tensor as T
from sklearn.feature_extraction import image
import nlinalg
import h5py
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from itertools import product

try:
    import PIL.Image as Image
except ImportError:
    import Image

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class dA(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

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
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
    # end-snippet-1

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        # Corruption Noise
        #noise = self.theano_rng.binomial(size=input.shape, n=1,
        #                                p=1 - corruption_level,
        #                                dtype=theano.config.floatX) * input

        #noise = self.theano_rng.normal(size=input.shape)*(corruption_level) + input
        noise = self.theano_rng.normal(avg=0.0,std=corruption_level,size=input.shape) + input

        return T.clip(noise,0,1)

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        # Implementation of cost function from the paper
        lambda_reg = 0.00001
        beta = 0.01
        rho = 0.2

        # ---- Error Term
        l2norm = T.sqrt(((self.x-z)**2).sum(axis=0,keepdims=False))**2
        errorterm = T.mean(l2norm)

        # ---- KL Divergence Term
        rho_j = T.mean(y,axis=0,keepdims=False) #Mean activation of hidden units based on hidden layer, results in 1 x HU matrix/vector
        kl = rho*T.log(rho/rho_j) + (1-rho)*T.log((1-rho)/(1-rho_j))
        kl = T.sum(kl)
        #T.sum((rho_expression),keepdims=False)

        # ---- Regularization Term
        regterm = (T.sqrt((self.W ** 2).sum())**2) + (T.sqrt((self.W_prime ** 2).sum())**2)

        # ---- Final Loss Function
        cost = errorterm + beta*kl + lambda_reg/2 * regterm

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

def reconstruct_from_patches_with_strides_2d(patches, image_size, strides):

    i_stride = strides[0]
    j_stride = strides[1]
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    img1 = np.zeros(image_size)
    n_h = int((i_h - p_h + i_stride)/i_stride)
    n_w = int((i_w - p_w + j_stride)/j_stride)
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i*i_stride:i*i_stride + p_h, j*j_stride:j*j_stride + p_w] +=p
        img1[i*i_stride:i*i_stride + p_h, j*j_stride:j*j_stride + p_w] +=np.ones(p.shape)
    return img/img1

""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


################## KIN MODIFIED THIS PART, DO NOT USE!!!!!! TRANSPOSED TILE OUTPUT ####
def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape).T)
                    else:
                        this_img = this_x.reshape(img_shape).T
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

def load_data(dataset):

    print '... loading h5py mat data'

    f = h5py.File(dataset)

    train_set_x = numpy.transpose(f['train_set_x'])
    valid_set_x = numpy.transpose(f['valid_set_x'])
    test_set_x = numpy.transpose(f['test_set_x'])
    train_set_y = numpy.transpose(f['train_set_y'])
    valid_set_y = numpy.transpose(f['valid_set_y'])
    test_set_y = numpy.transpose(f['test_set_y'])

    #train_set_x = train_set_x[0:train_set_x.shape[0]/100,:]
    #train_set_y = train_set_y[0:train_set_y.shape[0]/100,:]
    #valid_set_x = valid_set_x[0:valid_set_x.shape[0]/100,:]
    #valid_set_y = valid_set_y[0:valid_set_y.shape[0]/100,:]

    print train_set_x.shape
    print train_set_y.shape
    print valid_set_x.shape
    print valid_set_y.shape
    print test_set_x.shape
    print test_set_y.shape


    f.close()

    train_set = train_set_x, train_set_y
    valid_set = valid_set_x, valid_set_y
    test_set = test_set_x, test_set_y

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                       dtype=theano.config.floatX),
                       borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                       dtype=theano.config.floatX),
                       borrow=borrow)

        return shared_x, shared_y #T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_data_overlapped_strides(te_dataset, patch_size, strides):

    test_set_, te_h, te_w = overlapping_patches_strides(path=te_dataset, patch_size = patch_size, strides=strides)

    def shared_dataset(data_x, borrow=True):
        shared_data = theano.shared(numpy.asarray(data_x,
                          dtype=theano.config.floatX),
                          borrow=borrow)
        return shared_data

    test_set_ = shared_dataset(test_set_)
    rval = test_set_
    return rval, te_h, te_w


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out,W=None, b=None):

        if not W :
           self.W = theano.shared(
               value=numpy.zeros(
                   (n_in, n_out),
                   dtype=theano.config.floatX
               ),
               name='W',
               borrow=True
           )

        if not b:
           self.b = theano.shared(
               value=numpy.zeros(
                   (n_out,),
                   dtype=theano.config.floatX
               ),
               name='b',
               borrow=True
           )

        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        self.y_pred = self.p_y_given_x
        self.params = [self.W, self.b]

    def image_norm(self, y, obj):

        l2norm = ((y - self.y_pred)**2).sum(axis=1,keepdims=False)
        errorterm = T.mean(l2norm)

	lambda_reg = 0.00001

	weights = 0
	for i in xrange(obj.n_layers):
            #weight = (T.sqrt((obj.dA_layers[i].W ** 2).sum())**2)
            weight = (nlinalg.trace(T.dot(obj.dA_layers[i].W.T, obj.dA_layers[i].W))) #Frobenius norm
            weights = weights + weight
	regterm = T.sum(weights,keepdims=False)

	return T.mean(l2norm) + lambda_reg/2 *regterm

    def image_norm_noreg(self, y):

	y_diff = (y - self.y_pred)
	l2norm = (T.sqrt((y_diff**2).sum(axis=1,keepdims=False))**2)

	return T.mean(l2norm)

def   overlapping_patches_strides(path, patch_size, strides):

      Ols_images = Image.open (path).convert('RGB')
      height, width, channel = np.asarray(Ols_images).shape

      # ROC Pic
      nrow = height*1
      ncol = width*1

      #print '    ... Initial image dimensions: ', nrow, ncol

      Up = (nrow-patch_size[0]-strides[0])/strides[0]
      Vp = (ncol-patch_size[1]-strides[1])/strides[1]

      #print '    ... Initial patches: ', '%.2f'%(Up), '%.2f'%(Vp)

      Up = np.floor((nrow-patch_size[0]-strides[0])/strides[0])
      Vp = np.floor((ncol-patch_size[1]-strides[1])/strides[1])

      #print '    ... Generated patches: ', '%.2f'%(Up), '%.2f'%(Vp)

      nrow = np.int(Up*strides[0] + strides[0] + patch_size[0])
      ncol = np.int(Vp*strides[1] + strides[1] + patch_size[1])

      #print '    ... Resized image dimensions: ', nrow, ncol

      Ols_images = Ols_images.resize((ncol, nrow), Image.BICUBIC)
      Ols_images = np.asarray(Ols_images,dtype = 'float')/255

      #Ols_images = correlation.normalizeArray(np.asarray(Ols_images))
      #U = (nrow-patch_size[0]-strides[0])/strides[0]
      #V = (ncol-patch_size[1]-strides[1])/strides[1]

      image_height = np.asarray(Ols_images).shape[0]
      image_width = np.asarray(Ols_images).shape[1]

      print 'extracting color component', Ols_images.shape
      array_r = np.squeeze(Ols_images[:,:,0]);
      array_g = np.squeeze(Ols_images[:,:,1]);
      array_b = np.squeeze(Ols_images[:,:,2]);

      print 'processing R component'
      Ols_patche = image.extract_patches(array_r, patch_shape=patch_size, extraction_step=strides)
      Ols_patchesr = np.reshape(Ols_patche,(Ols_patche.shape[0]*Ols_patche.shape[1], -1))

      print 'processing G component'
      Ols_patche = image.extract_patches(array_g, patch_shape=patch_size, extraction_step=strides)
      Ols_patchesg = np.reshape(Ols_patche,(Ols_patche.shape[0]*Ols_patche.shape[1], -1))

      print 'processing B component'
      Ols_patche = image.extract_patches(array_b, patch_shape=patch_size, extraction_step=strides)
      Ols_patchesb = np.reshape(Ols_patche,(Ols_patche.shape[0]*Ols_patche.shape[1], -1))

      Ols_patches = np.concatenate((Ols_patchesr,Ols_patchesg,Ols_patchesb), axis=1)

      #n_patches, nvis = Ols_patches.shape
      rval = (Ols_patches, image_height, image_width)
      return rval