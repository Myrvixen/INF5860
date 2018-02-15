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
  #loss=[]
  #dw = []

  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = np.zeros(num_classes)
    for j in range(num_classes):
      scores[j] = np.dot(X[i], W[:,j])

    #Normalization to avoid numerical instability
    log_C = np.max(scores)
    scores -= log_C

    sum_scores = np.sum(np.exp(scores))
    label = y[i]
    loss += -np.log(np.exp(scores[label])/sum_scores)

    for j in range(num_classes):
        dW[:, j] += (np.exp(scores[j])/sum_scores - (j == label))*X[i]

  loss /= num_train
  dW /= num_train

  loss += 0.5*reg*np.sum(W**2)
  dW += reg*W

  scores = np.dot(X, W)


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
  #loss = []
  #dW = []

  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = np.dot(X, W)
  # Normalization for numerical stability
  scores -= np.max(scores, axis=1, keepdims=True)


  sum_scores = np.sum(np.exp(scores), axis=1, keepdims=True)
  p = np.exp(scores)/sum_scores

  # Pick out the probabilites for the correct class for every training sample
  correct_classes = p[range(num_train), y]

  # Calculating the total loss with regularization
  loss = np.sum(-np.log(correct_classes))/num_train
  loss += 0.5*reg*np.sum(W*W)


  # The indication function in the analytical expression of dW only results in the
  # probabilites for the correct classes being subtracted by 1
  p[range(num_train), y] -= 1

  dW = np.dot(X.T, p)/num_train
  dW += reg*W




  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
