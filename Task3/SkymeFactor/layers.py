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
    # TODO: Copy from previous assignment
    #raise Exception("Not implemented!")
    loss = reg_strength * (W**2).sum()
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
    predictions = np.array(predictions)
    p_dim = predictions.ndim
    if p_dim == 1:
      predictions = predictions[np.newaxis, :]
    predictions -= np.max(predictions, axis=1)[:, np.newaxis]
    
    probs = np.exp(predictions) / np.sum(np.exp(predictions), axis=1)[:, np.newaxis]

    if p_dim == 1:
      return probs[0]
    else:
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
    if (type(target_index) == int):
      loss = - np.log(probs)[target_index]
    else:
      loss = - np.mean(np.log(probs[range(target_index.shape[0]),target_index]))
    
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
    # TODO copy from the previous assignment
    #raise Exception("Not implemented!")
    zeros = np.zeros_like(predictions)
    if predictions.ndim > 1:  # batch case
      for i in range(target_index.shape[0]):
        zeros[i, target_index[i]] = 1
    else:
      zeros[target_index] = 1

    loss = cross_entropy_loss(softmax(predictions), target_index)
    grad = softmax(predictions)
    grad -= zeros

    if predictions.ndim > 1:
      grad /= predictions.shape[0]
    return loss.mean(), grad


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
        self.result = None
        #pass

    def forward(self, X):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")
        self.result = (X > 0)
        return self.result * X

    def backward(self, d_out):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")
        d_result = d_out * self.result#np.where(self.result > 0, d_out, 0)
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")
        self.X = X
        result = np.dot(self.X, self.W.value) + self.B.value

        return result

    def backward(self, d_out):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)

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
        self.X = None
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.stride = 1
        self.result = None


    def forward(self, X):
        p = self.padding
        self.X = np.pad(X, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant')
        batch_size, height, width, channels = X.shape

        out_height = round((height - self.filter_size + 2 * self.padding) / self.stride +1)
        out_width = round((width - self.filter_size + 2 * self.padding) / self.stride +1)
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        self.result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        #temp = np.array((batch_size, 1, 1, channels, 1))
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                #pass
                temp = (self.X[:, y:(y + self.filter_size), x:(x + self.filter_size), :, np.newaxis] * 
                    self.W.value[np.newaxis, :])
                temp = temp.sum(axis=(3, 2, 1))
                self.result[:, y, x] = temp + self.B.value

        return self.result
        #raise Exception("Not implemented!")


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        dx = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                #pass
                temp = self.X[:, y:(y + self.filter_size), x:(x + self.filter_size), :, np.newaxis]
                dout_grad = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :]
                self.W.grad += np.sum(dout_grad * temp, axis=0)
                dx[:, y:(y + self.filter_size), x:(x + self.filter_size), :] += np.sum(self.W.value * dout_grad, axis=4)
        
        self.B.grad += np.sum(d_out, axis=(0, 1, 2))

        p = self.padding
        height -= 2 * p
        width  -= 2 * p
        d_input = dx[:, p:(p + height), p:(p + width), :]

        return d_input
        #raise Exception("Not implemented!")

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
        batch_size, height, width, channels = X.shape
        self.X = X.copy()
        out_height = round((height - self.pool_size) / self.stride +1)
        out_width = round((width - self.pool_size) / self.stride +1)
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        #raise Exception("Not implemented!")
        result = np.zeros((batch_size, out_height, out_width, channels))
        for y in range(out_height):
            for x in range(out_width):
                temp = self.X[:, y:(y + self.pool_size), x:(x + self.pool_size), :]
                result[:, y, x] = temp.max(axis=(2, 1))

        return result


    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        #batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        #raise Exception("Not implemented!")
        result = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                temp = self.X[:, y:(y + self.pool_size), x:(x + self.pool_size), :]
                temp = (temp == temp.max(axis=(2, 1))[:, np.newaxis, np.newaxis, :])
                dx = (d_out[:, y, x, :])[:, np.newaxis, np.newaxis, :]
                result[:, y:(y + self.pool_size), x:(x + self.pool_size), :] += dx * temp

        return result

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = self.X_shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        #raise Exception("Not implemented!")
        
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        #raise Exception("Not implemented!")
        batch_size, height, width, channels = self.X_shape
        
        return d_out.reshape(batch_size, height, width, channels)

    def params(self):
        # No params!
        return {}
