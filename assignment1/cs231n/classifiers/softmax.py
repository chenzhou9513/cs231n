import numpy as np
from random import shuffle

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
  #############################################################################
  fxw = np.dot(X,W)    
  for i in range(X.shape[0]):
    fxw_i = fxw[i]-max(fxw[i])
    loss += -fxw_i[y[i]]+np.log(np.sum(np.exp(fxw_i)))
    for j in range(dW.shape[1]):
      if j==y[i]:  
        dW[:,j]+= -X[i]+(X[i]/np.log(np.sum(np.exp(fxw_i))))
      else:
        dW[:,j]+=X[i]/np.log(np.sum(np.exp(fxw_i)))
  
  loss /= X.shape[0] 
  loss +=  reg * np.sum(W * W)
  dW = dW/X.shape[0]  + 2*reg* W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  #############################################################################
  fxw = np.dot(X,W) 
  fxw = fxw - np.max(fxw,axis = 1).reshape(len(fxw),1)
  loss += np.sum(-fxw[list(range(len(fxw))),y] + np.log(np.sum(np.exp(fxw),axis=1)))
  loss /= X.shape[0] 
  loss +=  reg * np.sum(W * W)
  sum_f = np.sum(np.exp(fxw), axis=1, keepdims=True)
  p = np.exp(fxw)/sum_f
  ind = np.zeros_like(p)
  ind[np.arange(X.shape[0]), y] = 1
  dW = X.T.dot(p - ind)
  dW = dW/X.shape[0]  + 2*reg* W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

