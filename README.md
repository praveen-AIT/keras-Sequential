# keras-Sequential

The sequential API allows you to create models layer-by-layer for most problems. It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs.

Keras Sequential Models
-----------------------

This is a way of creating deep learning models where an instance of the Sequential class is created and model layers are created and added to it.

The Sequential model is a linear stack of layers.

You can create a Sequential model by passing a list of layer instances to the constructor:

# create first network with Keras
from keras.models import Sequential
from keras.layers import Dense

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

http://keras.dhpit.com/img/nn.png

# model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))

# It means 8 input parameters, with 12 neurons in the FIRST hidden layer.

# Moving on to Convolution1D and Convolution2D

Convolution1D

keras.layers.convolutional.Convolution1D(nb_filter, filter_length, init='glorot_uniform', activation=None, weights=None, border_mode='valid', subsample_length=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
C
onvolution operator for filtering neighborhoods of 1-D inputs.

When using this layer as the first layer in a model, either provide the keyword argument input_dim (int, e.g. 128 for sequences of 128-dimensional vectors), or input_shape (tuple of integers, e.g. (10, 128) for sequences of 10 vectors of 128-dimensional vectors).


# apply a convolution 1d of length 3 to a sequence with 10 timesteps,
# with 64 output filters
model = Sequential()
model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
# now model.output_shape == (None, 10, 64)

# add a new conv1d on top
model.add(Convolution1D(32, 3, border_mode='same'))
# now model.output_shape == (None, 10, 32)

Arguments

# nb_filter: Number of convolution kernels to use (dimensionality of the output).
# filter_length: The extension (spatial or temporal) of each filter.
# init: name of initialization function for the weights of the layer (see initializations), or alternatively, Theano function to use for weights initialization. This parameter is only relevant if you don't pass a weights argument.
# activation: name of activation function to use (see activations), or alternatively, elementwise Theano function. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
# weights: list of numpy arrays to set as initial weights.
# border_mode: 'valid', 'same' or 'full' ('full' requires the Theano backend).
# subsample_length: factor by which to subsample output.
# W_regularizer: instance of WeightRegularizer (eg. L1 or L2 regularization), applied to the main weights matrix.
# b_regularizer: instance of WeightRegularizer, applied to the bias.
# activity_regularizer: instance of ActivityRegularizer, applied to the network output.
# W_constraint: instance of the constraints module (eg. maxnorm, nonneg), applied to the main weights matrix.
# b_constraint: instance of the constraints module, applied to the bias.
# bias: whether to include a bias (i.e. make the layer affine rather than linear).
# input_dim: Number of channels/dimensions in the input. Either this argument or the keyword argument input_shapemust be provided when using this layer as the first layer in a model.
# input_length: Length of input sequences, when it is constant. This argument is required if you are going to connect  Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).

# Input shape

3D tensor with shape: (samples, steps, input_dim).

# Output shape

3D tensor with shape: (samples, new_steps, nb_filter). steps value might have changed due to padding.


# Now one important thing before moving on
# Role of “Flatten” in Keras

Dense(16, input_shape=(5,3))

would result in a Dense network with 3 inputs and 16 outputs which would be applied independently for each of 5 steps. So if D(x) transforms 3 dimensional vector to 16-d vector what you'll get as output from your layer would be a sequence of vectors: [D(x[0,:], D(x[1,:],..., D(x[4,:]] with shape (5, 16). In order to have the behaviour you specify you may first Flatten your input to a 15-d vector and then apply Dense:

model = Sequential()
model.add(Flatten(input_shape=(3, 2)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(4))
model.compile(loss='mean_squared_error', optimizer='SGD')

Images to make it understand better

https://i.stack.imgur.com/Wk8eV.png

Let's take one more example

model = Sequential()
model.add(Conv2D(64, (3, 3),
                 input_shape=(3, 32, 32), padding='same',))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())

The output shape now will be : model.output_shape == (None, 65536)

because due to flatten the output dimension would be (None, ). So to flatten the now 4D (64, 32, 32) all the remaining dimension would be clubbed as one by multiplying.
So, 64 X 32 X 32 = 65536.



