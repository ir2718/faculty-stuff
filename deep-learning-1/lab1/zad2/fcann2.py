import numpy as np
import matplotlib.pyplot as plt
from data import *

def relu(x):
    return np.maximum(0., x)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=1).reshape(-1, 1)

def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted, axis=1).reshape(-1, 1)
    return probs

def loss_f(y, y_hat, W1, W2, param_lambda):
    loss = -np.mean(np.sum(y*np.log(y_hat), axis=1))
    regularization = param_lambda * (np.sum(W1**2) + np.sum(W2**2))
    return loss + regularization

def fcann2_train(X, y):
    hidden_layer_dim = 50
    num_classes = len(set(y))

    y_encoded = class_to_onehot(y)

    W1, b1 = np.random.randn(X.shape[1], hidden_layer_dim), np.zeros((1, hidden_layer_dim))
    W2, b2 = np.random.randn(hidden_layer_dim, num_classes), np.zeros((1, num_classes))

    param_niter = 1000
    param_delta = 0.00003
    param_lambda = 1e-3

    for i in range(param_niter):
        # -------------------------
        # forward pass
        s1 = np.dot(X, W1) + b1
        h1 = relu(s1)

        s2 = np.dot(h1, W2) + b2
        h2 = stable_softmax(s2)

        # ------------------------
        # loss calculation
        loss = loss_f(y_encoded, h2, W1, W2, param_lambda)
        print(f'step {i} --- loss: {loss}')

        # ------------------------
        # backward pass - last layer
        gs2 = h2 - y_encoded                                         # N x C
 
        grad_W2 = np.dot(gs2.T, h1) + param_lambda * 2*W2.T          # C x H
        grad_b2 = np.sum(gs2, axis=0).reshape(-1, 1)                 # C x 1

        W2 += -param_delta * grad_W2.T
        b2 += -param_delta * grad_b2.T

        # backward pass - first layer
        gs1 = (np.diag(s1)>0).reshape(-1, 1).T * np.dot(gs2, W2.T)   # N x H

        grad_W1 = np.dot(gs1.T, X) + param_lambda * 2*W1.T           # H x D
        grad_b1 = np.sum(gs1, axis=0).reshape(-1, 1)                 # H x 1

        W1 += -param_delta * grad_W1.T
        b1 += -param_delta * grad_b1.T
        # ----------------------

    return W1, b1, W2, b2
    
def fcann2_classify(X, W1, b1, W2, b2):
    s1 = np.dot(X, W1) + b1
    h1 = relu(s1)

    s2 = np.dot(h1, W2) + b2
    h2 = stable_softmax(s2)

    return np.argmax(h2, axis=1)
    

def main():
    np.random.seed(100)

    # X, y = sample_gmm_2d(K=6, C=2, N=10)
    # X, y = sample_gmm_2d(K=2, C=2, N=100)
    X, y = sample_gmm_2d(K=3, C=3, N=100)
    
    W1, b1, W2, b2 = fcann2_train(X, y)
    preds = fcann2_classify(X, W1, b1, W2, b2)
    eval_perf_multi(y, preds)

    decfun = lambda X: fcann2_classify(X, W1, b1, W2, b2)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0.5)
    
    graph_data(X, y, preds)
    plt.show()

main()