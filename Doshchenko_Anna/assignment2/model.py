import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.fully_connect_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.fully_connect_2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        params = self.params()
        params['W1'].grad = np.zeros_like(params['W1'].value)
        params['W2'].grad = np.zeros_like(params['W2'].value)
        params['B1'].grad = np.zeros_like(params['B1'].value)
        params['B2'].grad = np.zeros_like(params['B2'].value)
        
        fully_connect_out_1 = self.fully_connect_1.forward(X)
        relu_out = self.relu.forward(fully_connect_out_1)
        fully_connect_out_2 = self.fully_connect_2.forward(relu_out)
        
        loss, d_preds = softmax_with_cross_entropy(fully_connect_out_2, y)
        
        dfully_connect_out_2 = self.fully_connect_2.backward(d_preds)
        drelu_out = self.relu.backward(dfully_connect_out_2)
        fully_connect_out_1 = self.fully_connect_1.backward(drelu_out)
        
        for param in params:
            l2_loss, grad = l2_regularization(params[param].value, self.reg)
            loss += l2_loss
            params[param].grad += grad
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        res1 = self.fully_connect_1.forward(X)
        res2 = self.relu.forward(res1)
        res3 = self.fully_connect_2.forward(res2)
        pred = np.argmax(res3, axis = 1)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        result['W1'] = self.fully_connect_1.params()['W']
        result['W2'] = self.fully_connect_2.params()['W']
        result['B1'] = self.fully_connect_1.params()['B']
        result['B2'] = self.fully_connect_2.params()['B']

        return result
