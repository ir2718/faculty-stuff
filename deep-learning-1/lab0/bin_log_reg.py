import numpy as np
import data 
import matplotlib.pyplot as plt

def binlogreg_train(X, Y_):
    '''
    Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

    Povratne vrijednosti
        w, b: parametri logističke regresije
    '''
    param_niter = 3000
    param_delta = 0.05

    w, b = np.random.randn(X.shape[1], 1), 0
    # print(w, b)

    len_ = len(Y_)
    Y_ = Y_.reshape(-1, 1)
    # print(len_)

    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter):
        # klasifikacijske mjere
        scores = np.dot(X, w) + b    # N x 1
        # print(scores.shape)

        # vjerojatnosti razreda c_1
        probs = 1/(1 + np.exp(-scores))     # N x 1
        # print(probs.shape)

        # gubitak
        # loss  = -1/len_ * np.sum(np.multiply(Y_, np.log(probs)) + np.multiply(1 - Y_, np.log(probs)))     # scalar
        loss = -1/len_ * np.sum(np.log([x if x > 0.5 else 1-x for x in probs]))
        # print(loss.shape)

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskim mjerama
        dL_dscores = probs - Y_     # N x 1
        # print(dL_dscores)

        # gradijenti parametara
        grad_w = 1/len_ * np.dot(dL_dscores.T, X).T      # D x 1
        grad_b = 1/len_ * np.sum(dL_dscores)     # 1 x 1

        # poboljšani parametri
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def binlogreg_classify(X, w, b):
    '''
    Argumenti
        X:    podatci, np.array NxD
        w, b: parametri logističke regresije 

    Povratne vrijednosti
        probs: vjerojatnosti razreda c1
    '''
    scores = np.dot(X, w) + b
    return (1/(1 + np.exp(-scores))).reshape(-1)

def binlogreg_decfun(w,b):
    return lambda X: binlogreg_classify(X, w, b)

if __name__ == '__main__':
    # instantiate the dataset
    X,Y_ = data.sample_gauss_2d(2, 100)

    # train the logistic regression model
    w,b = binlogreg_train(X, Y_)

    # evaluate the model on the train set
    probs = binlogreg_classify(X, w, b)

    # recover the predicted classes Y
    Y = probs > 0.5

    # evaluate and print performance measures
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)

    # graph the decision surface
    decfun = binlogreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y)

    # show the plot
    plt.show()

