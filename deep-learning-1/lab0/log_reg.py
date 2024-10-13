import numpy as np
import matplotlib.pyplot as plt
import data

def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs

def logreg_train(X, Y_):
    param_niter = 5000
    param_delta = 0.001

    len_ = X.shape[0]
    C, D = len(set(Y_)), X.shape[1] 
    W, b = np.random.randn(C, D), np.zeros(shape=(C, 1))
    
    n_values = np.max(Y_) + 1
    Y = np.eye(n_values)[Y_]
    # print(W.shape, b.shape)

    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter):
        # eksponencirane klasifikacijske mjere
        scores = (np.dot(W, X.T) + b).T    # N x C
        # print(scores.shape)
        expscores = np.exp(scores) # N x C
        # print(expscores.shape)
        
        # nazivnik sofmaksa
        sumexp = np.sum(expscores, axis=1).reshape(-1, 1)   # N x 1
        # print(sumexp.shape)

        # logaritmirane vjerojatnosti razreda 
        probs = expscores/sumexp     # N x C
        # print(probs.shape)
        logprobs = np.log(probs)  # N x C

        # gubitak
        loss = -1/len_ * np.sum(logprobs) # scalar
        # print(loss.shape)

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        dL_ds = probs - Y     # N x C
        # print(dL_ds.shape)

        # gradijenti parametara
        grad_W = 1/len_ * np.dot(dL_ds.T, X)    # C x D (ili D x C)
        grad_b = 1/len_ * np.sum(dL_ds.T, axis=1).reshape(-1, 1)    # C x 1 (ili 1 x C)
        # print(grad_W.shape, grad_b.shape)

        # poboljšani parametri
        W += -param_delta * grad_W
        b += -param_delta * grad_b
        # print('----------------')
    
    return W, b

def eval_perf_multi(Y, Y_):
    conf_matrix = []
    values = set(Y)
    for v in values:
        tmp = []
        for v2 in values:
            y_pred = tuple(np.array(np.where(Y  ==  v)).tolist()[0])
            y_true = tuple(np.array(np.where(Y_ == v2)).tolist()[0])

            if y_true == [] or y_pred == []:
                tmp.append(0)
            else:
                final = set(y_pred) & set(y_true)
                tmp.append(len(final))
        conf_matrix.append(tmp)

    diag = np.diagonal(np.array(conf_matrix))
    acc  = np.sum(diag)/np.sum(conf_matrix)
    pr = diag/np.sum(conf_matrix, axis=0)
    re = diag/np.sum(conf_matrix, axis=1)
    
    return acc, conf_matrix, pr, re


def logreg_classify_(X, W, b):
    scores = (np.dot(W, X.T) + b).T
    return np.argmax(scores, axis=1)

def logreg_classify(W, b):
    return lambda X: logreg_classify_(X, W, b)

from sklearn.metrics import confusion_matrix, recall_score, precision_score

if __name__ == '__main__':
    # instantiate the dataset
    X,Y_ = data.sample_gauss_2d(3, 100)

    # train the logistic regression model
    W,b = logreg_train(X, Y_)

    # evaluate the model on the train set
    Y = logreg_classify_(X, W, b)

    # evaluate and print performance measures
    accuracy, conf_matrix, recall, precision = eval_perf_multi(Y, Y_)
    print(accuracy)
    print(conf_matrix)
    print(recall)
    print(precision)

    # graph the decision surface
    decfun = logreg_classify(W, b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y)

    # show the plot
    plt.show()