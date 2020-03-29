import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss =reg_strength * (W**2).sum()
    grad = 2 * reg_strength * W
    return loss, grad

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    probs = predictions.copy()
    if predictions.ndim == 1:
        probs -= np.max(predictions)
        probs = np.exp(probs)/np.sum(np.exp(probs))
    else:
        probs -= np.max(predictions, axis = 1).reshape(-1, 1)
        probs = np.exp(probs) / np.sum(np.exp(probs), axis=1).reshape(-1, 1)
    return probs

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        loss = - sum(np.log(probs[x, target_index[x]]) for x in range(probs.shape[0]))
        loss /= batch_size
    return loss

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()
    if dprediction.ndim == 1:
        dprediction[target_index] -= 1
    else:
        flat_array = target_index.flatten()
        batch_size = probs.shape[0]
        array = np.arange(batch_size)
        dprediction[array, flat_array] -= 1
        dprediction /= batch_size
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        ind = (X > 0)
        return X * ind

    def backward(self, d_out):
        ind = (self.X >= 0)
        d_result = d_out * ind
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        result = np.dot(X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        self.B.grad += np.sum(d_out, axis = 0)

        self.W.grad += np.dot(self.X.T, d_out)
        
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        padding = self.padding
        filter_size, filter_size, input_channels, output_channels = self.W.value.shape
        W = self.W.value.reshape((filter_size * filter_size * input_channels, output_channels))
        self.X = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
        out_height = height - filter_size + 2*padding +1
        out_width = width - filter_size + 2*padding + 1
        result = np.zeros((batch_size, out_height, out_width, output_channels))
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                window = self.X[:, y : y + filter_size, x : x + filter_size, :]
                window = window.reshape(batch_size, filter_size*filter_size*input_channels)
                result[:, y, x] = window @ W+ self.B.value
                # TODO: Implement forward pass for specific location
        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # we make a reshape for easier calculations later
        filter_size, filter_size, input_channels, output_channels = self.W.value.shape
        res = np.zeros_like(self.X)
        W = self.W.value.reshape((filter_size * filter_size * input_channels, output_channels))
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                window = self.X[:, y : y + filter_size, x : x + filter_size, :]
                window = window.reshape((batch_size, filter_size*filter_size*input_channels))
                back = d_out[:, y, x, :] @ W.T
                back = back.reshape((batch_size, filter_size, filter_size, input_channels))
                res[:, y : y + filter_size, x : x + filter_size, :] += back
                grad = window.T @d_out[:, y, x, :]
                grad = grad.reshape(self.W.value.shape)
                self.W.grad += grad
                self.B.grad += np.sum(d_out[:, y, x, :], axis=0)
        return res[:, self.padding:height-self.padding, self.padding:width-self.padding, :]


    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        pool_size = self.pool_size
        stride = self.stride
        out_height, out_width = round((height - pool_size)/stride) + 1, round((width - pool_size)/stride) + 1 
        result = np.zeros((batch_size, out_height, out_width, channels))
        for y in range(out_height):
            for x in range(out_width):
                window = X[:, y : y + pool_size,  x : x + pool_size, :]
                result[:, y, x, :] = np.max(window, axis=(1, 2))
        return result


    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        result_out = np.zeros_like(self.X)
        pool_size = self.pool_size
        stride = self.stride
        _, out_height, out_width, _ = d_out.shape
        for y in range(out_height):
            for x in range(out_width):
                window = self.X[:, y: y + pool_size,  x : x + pool_size, :]
                grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                window = (window == window.max(axis=(2, 1))[:, np.newaxis, np.newaxis, :])
                result_out[:, y : y + pool_size, x :  x + pool_size, :] += grad * window
        return result_out

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        result = X.reshape(batch_size, height*width*channels)
        return result

    def backward(self, d_out):
        # TODO: Implement backward pass
        batch_size, height, width, channels = self.X_shape
        result = d_out.reshape(batch_size, height, width, channels)
        return result

    def params(self):
        # No params!
        return {}
