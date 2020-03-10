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
        # TODO Create necessary layers
        self.first_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.first_relu = ReLULayer()
        self.second_layer = FullyConnectedLayer(hidden_layer_size, n_output)

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
        for i, w in self.params().items():
            w.grad = np.zeros(w.grad.shape)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        value = self.first_layer.forward(X)
        value = self.first_relu.forward(value)
        value = self.second_layer.forward(value)
        loss, grads = softmax_with_cross_entropy(value, y)
        value = self.second_layer.backward(grads)
        value = self.first_relu.backward(value)
        value = self.first_layer.backward(value)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for i, w in self.params().items():
            loss_delta, grad_delta = l2_regularization(w.value, self.reg)
            w.grad += grad_delta
            loss += loss_delta

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
        
        pred = self.first_layer.forward(X)
        pred = self.first_relu.forward(pred)
        pred = self.second_layer.forward(pred)
        pred = np.argmax(pred, axis=1)
        
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        result['first_layer_W'] = self.first_layer.params()['W']
        result['second_layer_W'] = self.second_layer.params()['W']
        result['first_layer_B'] = self.first_layer.params()['B']
        result['second_layer_B'] = self.second_layer.params()['B']

        return result
