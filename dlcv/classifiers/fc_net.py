import numpy as np
from dlcv.layers import *
class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be
    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """
 
    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
         dropout_keep_ratio=0.5,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
 
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        for i in range(self.num_layers):
          if i == 0:
            self.params['W1'] = np.random.randn(input_dim, hidden_dims[i]) * weight_scale
            self.params['b1'] = np.zeros(hidden_dims[i])
            if self.normalization == "batchnorm":
              self.params['gamma1'] = np.random.randn(hidden_dims[i])
              self.params['beta1'] = np.random.randn(hidden_dims[i])
          elif i == self.num_layers -1:
            self.params['W'+str(i+1)] = np.random.randn(hidden_dims[i-1], num_classes) * weight_scale
            self.params['b'+str(i+1)] = np.zeros(num_classes)
          else:
            self.params['W'+str(i+1)] = np.random.randn(hidden_dims[i-1], hidden_dims[i]) * weight_scale
            self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])
            if self.normalization == "batchnorm":
              self.params['gamma'+str(i+1)] = np.random.randn(hidden_dims[i])
              self.params['beta'+str(i+1)] = np.random.randn(hidden_dims[i])
 
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
 
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed
 
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]
 
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
 
    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"
 
        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x = X.copy()
        #cache = []
        #cache_relu = []
        #{affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        for i in range(self.num_layers - 1):
          w = self.params['W'+str(i+1)]
          b = self.params['b'+str(i+1)]
          x, cache_temp = affine_forward(x, w, b)
          #cache.append(cache_temp)
          if self.normalization == "batchnorm":
            gamma = self.params['gamma'+str(i+1)]
            beta = self.params['beta'+str(i+1)]
            bn_param = self.bn_params[i]
            x, cache_temp = batchnorm_forward(x, gamma, beta, bn_param)
          x, cache_temp = relu_forward(x)
          #cache_relu.append(cache_temp)
        w = self.params['W'+str(self.num_layers)]
        b = self.params['b'+str(self.num_layers)]
        scores, cache_temp = affine_forward(x, w, b)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
 
        # If test mode return early
        if mode == "test":
            return scores
 
        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 
        #do forward propagation
        x = X.copy()
        cache = []
        cache_bp = []
        cache_relu = []
        cache_dropout = []
        #{affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        for i in range(self.num_layers - 1):
          w = self.params['W'+str(i+1)]
          b = self.params['b'+str(i+1)]
          x, cache_temp = affine_forward(x, w, b)
          cache.append(cache_temp)
          if self.normalization == "batchnorm":
            gamma = self.params['gamma'+str(i+1)]
            beta = self.params['beta'+str(i+1)]
            bn_param = self.bn_params[i]
            x, cache_temp = batchnorm_forward(x, gamma, beta, bn_param)
            cache_bp.append(cache_temp)
          x, cache_temp = relu_forward(x)
          cache_relu.append(cache_temp)
          if self.use_dropout:
            x, cache_temp = dropout_forward(x, self.dropout_param)
            cache_dropout.append(cache_temp)
        w = self.params['W'+str(self.num_layers)]
        b = self.params['b'+str(self.num_layers)]
        scores, cache_temp = affine_forward(x, w, b)
        cache.append(cache_temp)
        loss, dscores = softmax_loss(scores, y)
        for i in range(self.num_layers):
          w = self.params['W'+str(i+1)]
          loss += 0.5 * self.reg * np.sum(w * w)
        #do backward propagation
        dx, dw, db = affine_backward(dscores, cache.pop())
        grads['W'+str(self.num_layers)] = dw + self.reg * self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db
        for i in range(self.num_layers - 1)[::-1]:
          if self.use_dropout:
            dx = dropout_backward(dx, cache_dropout.pop())
          dx = relu_backward(dx, cache_relu.pop())
          if self.normalization == "batchnorm":
            dx, dgamma, dbeta = batchnorm_backward_alt(dx, cache_bp.pop())
            grads['gamma'+str(i+1)] = dgamma 
            grads['beta'+str(i+1)] = dbeta 
          dx, dw, db = affine_backward(dx, cache.pop())
          grads['W'+str(i+1)] = dw + self.reg * self.params['W'+str(i+1)]
          grads['b'+str(i+1)] = db
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
 
        return loss, grads
    def dropout_forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.

        Inputs:
        - x: Input data, of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We keep each neuron output with probability p.
          - mode: 'test' or 'train'. If the mode is train, then perform dropout;
            if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed makes this
            function deterministic, which is needed for gradient checking but not
            in real networks.

        Outputs:
        - out: Array of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
          mask that was used to multiply the input; in test mode, mask is None.

        NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.

        NOTE 2: Keep in mind that p is the probability of **keep** a neuron
        output; this might be contrary to some sources, where it is referred to
        as the probability of dropping a neuron output.
        """
        p, mode = dropout_param["p"], dropout_param["mode"]
        if "seed" in dropout_param:
            np.random.seed(dropout_param["seed"])

        mask = None
        out = None

        if mode == "train":
            #######################################################################
            # TODO: Implement training phase forward pass for inverted dropout.   #
            # Store the dropout mask in the mask variable.                        #
            # TODO: 实施训练阶段的前向传播，反向dropout，存储在掩码变量中。
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            mask = np.random.randn(*x.shape) < (1-p)
            mask = mask/(1-p)
            out = mask*x
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #######################################################################
            #                           END OF YOUR CODE                          #
            #######################################################################
        elif mode == "test":
            #######################################################################
            # TODO: Implement the test phase forward pass for inverted dropout.   #
            # TODO: 实现反向dropout的测试阶段正向传播。
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            out = x

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #######################################################################
            #                            END OF YOUR CODE                         #
            #######################################################################

        cache = (dropout_param, mask)
        out = out.astype(x.dtype, copy=False)

        return out, cache


    def dropout_backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.

        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from dropout_forward.
        """
        dropout_param, mask = cache
        mode = dropout_param["mode"]

        dx = None
        if mode == "train":
            #######################################################################
            # TODO: Implement training phase backward pass for inverted dropout   #
            # TODO: 实现反向dropout训练阶段的反向传递
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            dx = dout*mask

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #######################################################################
            #                          END OF YOUR CODE                           #
            #######################################################################
        elif mode == "test":
            dx = dout
        return dx