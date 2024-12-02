from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    N = X.shape[0]
    C = W.shape[1]
    D = W.shape[0]
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    m=np.zeros((N,C))
    grad_w3=np.zeros((D,C))
    dic={}
    for i in range(num_train):
        grad_temp=np.zeros((D,C))
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            
            if margin > 0:
                dW[:,j]+=X[i].transpose()
                dW[:,y[i]]+=-X[i].transpose()
                grad_w3[:,j]+=X[i].transpose()
                grad_w3[:,y[i]]+=-X[i].transpose()
                m[i, j] = 1
                loss += margin
        dic[i]=grad_temp/num_train

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW/=num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW+=reg*W*2
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    # TODO:
    # 计算损失函数的梯度并将其存储为dW。
    # 而不是先计算损失，再计算导数，
    # 在计算损失的同时计算导数可能更简单。因此，你可能需要修改上面的一些代码来计算梯度。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 梯度同样处理
    dW /= num_train
    # 正则项的梯度
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    # TODO:
    # 实现矢量化版本的结构化SVM损失, 将结果存储在损失。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X @ W
    scores_correct = scores[range(num_train),y].reshape((scores.shape[0],1))
    margins = np.maximum(0,scores - scores_correct + 1)
    margins[range(num_train),y] = 0
    loss += np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    # TODO:
    # 为结构化SVM损失实现梯度的矢量化版本，将结果存储在数据仓库中。
    #
    # 提示: 与其从头开始计算梯度，不如重用一些用于计算损失的中间值。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margins[margins > 0] = 1
    row_sum = np.sum(margins,axis = 1)
    margins[range(num_train),y] = -row_sum
    dW += np.dot(X.T, margins)/num_train + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
