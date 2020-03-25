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
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
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
        # TODO copy from the previous assignment
        self.grad = np.zeros(X.shape)
        self.grad[X > 0] = 1
        self.grad[X == 0] = 0.5
        X[X < 0] = 0
        return X

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = self.grad * d_out
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
        self.X = X.copy()
        res = X @ self.W.value
        res += self.B.value
        return res

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        self.W.grad = self.X.T @ d_out
        self.B.grad = np.sum(d_out, axis=0).reshape(self.B.grad.shape[0], -1)
        d_input = d_out @ self.W.value.T
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

        out_height = (height + 2 * self.padding) - self.filter_size + 1
        out_width = (width + 2 * self.padding) - self.filter_size + 1
        
        in_tensor = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, channels))
        
        in_tensor[:,self.padding:height+self.padding,self.padding:width+self.padding,:] = X
        self.X = in_tensor
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        out_tensor = np.zeros((batch_size, out_height, out_width, self.out_channels))
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                temp = (in_tensor[:,y:y+self.filter_size,x:x+self.filter_size,:]\
                                       .reshape((batch_size, self.filter_size**2*channels))\
                                       @ self.W.value.reshape((self.filter_size**2*channels, self.out_channels)) + \
                                       self.B.value).reshape((batch_size, self.out_channels))
                out_tensor[:,y,x,:] = temp
                # TODO: Implement forward pass for specific location
        return out_tensor


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
        d_input = np.zeros((batch_size, height, width, channels))
        self.W.grad = np.zeros((self.filter_size, self.filter_size, channels, out_channels))
        self.B.grad = np.zeros((out_channels))

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                self.W.grad += (self.X[:,y:y+self.filter_size,x:x+self.filter_size,:]\
                                .reshape((batch_size, self.filter_size**2*channels)).T @ \
                                d_out[:,y,x,:].reshape((batch_size, out_channels)))\
                                .reshape(self.W.grad.shape)
                self.B.grad += np.sum(d_out[:,y,x,:], axis=0).flatten()
        
        for y in range(out_height):
            for x in range(out_width):
                d_input[:,y:y+self.filter_size,x:x+self.filter_size,:]+=(d_out[:,y,x,:].reshape((batch_size, out_channels)) @\
                                    self.W.value.reshape((self.filter_size**2*channels, out_channels)).T)\
                                    .reshape((batch_size, self.filter_size, self.filter_size, channels))

        return d_input[:,self.padding:d_input.shape[1]-self.padding,self.padding:d_input.shape[2]-self.padding,:]

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
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X.copy()
        result = np.zeros((batch_size, width // self.stride, height // self.stride, channels))
        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                result[:,y // self.stride,x // self.stride,:] = np.amax(X[:,y:min(y+self.pool_size,height),x:min(x+self.pool_size,width),:], axis=(1,2))
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        d_input = np.zeros(self.X.shape)
        batch_es = np.repeat(np.arange(batch_size), channels)
        channel_s = np.tile(np.arange(channels), batch_size)
        
        for y in range(d_out.shape[1]):
            for x in range(d_out.shape[2]):
                max_ids = np.argmax(\
                        self.X[:,y * self.stride:y * self.stride + self.pool_size,x * self.stride:x * self.stride + self.pool_size,:].reshape((batch_size,-1,channels)),axis=1)
                temp = d_input[:,y * self.stride:y * self.stride + self.pool_size,x * self.stride:x * self.stride + self.pool_size,:].reshape((batch_size, -1, channels))
                temp[batch_es, max_ids.flatten(), channel_s] = d_out[batch_es,y,x,channel_s]
                d_input[:,y*self.stride:y*self.stride+self.pool_size,x*self.stride:x*self.stride+self.pool_size,:] =\
                        temp.reshape((batch_size, self.pool_size, self.pool_size, channels))
        return d_input
        

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape((batch_size, -1))

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
