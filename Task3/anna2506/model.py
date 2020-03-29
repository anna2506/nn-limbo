import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        image_width, image_height, n_channels = input_shape
        self.conv_1 = ConvolutionalLayer(n_channels, conv1_channels, 3, 1)
        self.relu_1 = ReLULayer()
        self.max_pool_1 = MaxPoolingLayer(4, 4)
        self.conv_2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.relu_2 = ReLULayer()
        self.max_pool_2 = MaxPoolingLayer(4, 4)
        self.flatten = Flattener()
        self.fully_connected = FullyConnectedLayer(4 * conv2_channels, n_output_classes)
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        params = self.params()
        params['Wconv1'].grad = np.zeros_like(params['Wconv1'].value)
        params['Wconv2'].grad =  np.zeros_like(params['Wconv2'].value)
        params['Wfc'].grad = np.zeros_like(params['Wfc'].value)
        params['Bconv1'].grad = np.zeros_like(params['Bconv1'].value)
        params['Bconv2'].grad = np.zeros_like(params['Bconv2'].value)
        params['Bfc'].grad = np.zeros_like(params['Bfc'].value)

        res1 = self.conv_1.forward(X)
        res2 = self.relu_1.forward(res1)
        res3 = self.max_pool_1.forward(res2)
        res4 = self.conv_2.forward(res3)
        res5 = self.relu_2.forward(res4)
        res6 = self.max_pool_2.forward(res5)
        res7 = self.flatten.forward(res6)
        res8 = self.fully_connected.forward(res7)

        loss, d_preds = softmax_with_cross_entropy(res8, y)
        
        dres1 = self.fully_connected.backward(d_preds)
        dres2 = self.flatten.backward(dres1)
        dres3 = self.max_pool_2.backward(dres2)
        dres4 = self.relu_2.backward(dres3)
        dres5 = self.conv_2.backward(dres4)
        dres6 = self.max_pool_1.backward(dres5)
        dres7 = self.relu_1.backward(dres6)
        dres8 = self.conv_1.backward(dres7)
        
        
        
        return loss


    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = np.zeros(X.shape[0], np.int)
        res1 = self.conv_1.forward(X)
        res2 = self.relu_1.forward(res1)
        res3 = self.max_pool_1.forward(res2)
        res4 = self.conv_2.forward(res3)
        res5 = self.relu_2.forward(res4)
        res6 = self.max_pool_2.forward(res5)
        res7 = self.flatten.forward(res6)
        res8 = self.fully_connected.forward(res7)
        pred = np.argmax(res8, axis = 1)
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        result['Wconv1'] = self.conv_1.W
        result['Wconv2'] = self.conv_2.W
        result['Wfc'] = self.fully_connected.W
        result['Bconv1'] = self.conv_1.B
        result['Bconv2'] = self.conv_2.B
        result['Bfc'] = self.fully_connected.B

        return result
