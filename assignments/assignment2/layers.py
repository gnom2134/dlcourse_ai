import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
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
    """
    # TODO: Copy from the previous assignment
    target_index = np.array(target_index)
    
    if len(preds.shape) == 1:
        probs = np.array(preds).reshape(1, preds.shape[0])
    else:
        probs = np.array(preds)
    probs = np.subtract(probs.T, np.max(probs, axis=1)).T
    probs = np.exp(probs)
    probs = np.divide(probs.T, np.sum(probs, axis=1)).T
    
    add_arr = np.zeros(probs.shape)
    add_arr[range(add_arr.shape[0]), target_index.flatten()] = 1
    loss = np.sum(-1 * add_arr * np.log(probs))
    
    d_preds = probs
    d_preds[range(d_preds.shape[0]), target_index.flatten()] -= 1

    return loss, d_preds.reshape(preds.shape)


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.grad = np.zeros(X.shape)
        self.grad[X > 0] = 1
        self.grad[X == 0] = 0.5
        X[X < 0] = 0
        return X

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = self.grad * d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        res = X @ self.W.value
        res += self.B.value
        return res

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        self.W.grad = self.X.T @ d_out
        self.B.grad = np.sum(d_out, axis=0).reshape(self.B.grad.shape[0], -1)
        d_input = d_out @ self.W.value.T 
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
