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
        # TODO Create necessary layers
        #raise Exception("Not implemented!")
        self.l_conv1 = ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1)
        self.l_Relu1 = ReLULayer()
        self.l_MxPl1 = MaxPoolingLayer(4, 4)
        self.l_conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.l_Relu2 = ReLULayer()
        self.l_MxPl2 = MaxPoolingLayer(4, 4)
        self.l_flat = Flattener()
        self.l_FC = FullyConnectedLayer(4 * conv2_channels, n_output_classes)



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
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        #raise Exception("Not implemented!")
        pred = self.l_conv1.forward(X)
        pred = self.l_Relu1.forward(pred)
        pred = self.l_MxPl1.forward(pred)
        pred = self.l_conv2.forward(pred)
        pred = self.l_Relu2.forward(pred)
        pred = self.l_MxPl2.forward(pred)
        pred = self.l_flat.forward(pred)
        pred = self.l_FC.forward(pred)
        loss, loss_grad = softmax_with_cross_entropy(pred, y)
        
        grad = self.l_FC.backward(loss_grad)
        grad = self.l_flat.backward(grad)
        grad = self.l_MxPl2.backward(grad)
        grad = self.l_Relu2.backward(grad)
        grad = self.l_conv2.backward(grad)
        grad = self.l_MxPl1.backward(grad)
        grad = self.l_Relu1.backward(grad)
        grad = self.l_conv1.backward(grad)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        #raise Exception("Not implemented!")
        pred = self.l_conv1.forward(X)
        pred = self.l_Relu1.forward(pred)
        pred = self.l_MxPl1.forward(pred)
        pred = self.l_conv2.forward(pred)
        pred = self.l_Relu2.forward(pred)
        pred = self.l_MxPl2.forward(pred)
        pred = self.l_flat.forward(pred)
        pred = self.l_FC.forward(pred)
        pred = np.argmax(pred, axis=1)

        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        #raise Exception("Not implemented!")
        result['Conv1W'] = self.l_conv1.params()['W']
        result['Conv2W'] = self.l_conv2.params()['W']
        result['FC_W'] = self.l_FC.params()['W']
        result['Conv1B'] = self.l_conv1.params()['B']
        result['Conv2B'] = self.l_conv2.params()['B']
        result['FC_B'] = self.l_FC.params()['B']

        return result
