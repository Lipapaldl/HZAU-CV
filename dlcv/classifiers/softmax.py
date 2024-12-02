from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    # TODO: 使用显式循环计算softmax损失及其梯度。
    # 将损失存储在loss中，梯度存储在dW中。如果你在这里不小心，很容易遇到数值不稳定。不要忘记正规化!
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    nt = X.shape[0] #训练集大小
    nc = W.shape[1] #类别
    for i in range(nt):
        s = np.exp(X[i] @ W)
        p = s / np.sum(s)
        loss += -np.log(p[y[i]])
        #梯度求解
        for j in range(nc):
            p_j = p[j]
            if j == y[i]:
                dW[:,j] += (p_j - 1) * X[i]
            else:
                dW[:,j] += p_j * X[i]
    loss /= nt
    dW /= nt
    loss += 0.5 *reg * np.sum(W*W)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    # TODO: 不使用显式循环计算softmax损失及其梯度。将损失存储在loss中，梯度存储在dW中。
    # 如果你在这里不小心，很容易遇到数值不稳定。别忘了正则化!
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n_t = X.shape[0]
    s = X @ W  # 计算分数矩阵。 
    exp_s = np.exp(s)  
    sum_s = np.sum(exp_s, axis = 1) 
    exp_s /= sum_s[:,np.newaxis]  # 标准化后
    loss_matrix = -np.log(exp_s[range(n_t),y])  
    loss += np.sum(loss_matrix)
    exp_s[range(n_t),y] -= 1  # 取正确标签处，减一
    dW += X.T @ exp_s
    #正则化
    loss /= n_t
    loss += reg * np.sum(W*W)
    dW /= n_t
    dW +=reg *W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
