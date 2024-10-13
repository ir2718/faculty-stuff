import torch
import torch.nn as nn
import numpy as np
from data import *
import matplotlib.pyplot as plt

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
            - D: dimensions of each datapoint 
            - C: number of classes
        """
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...
        super().__init__()
        self.W = nn.Parameter(torch.normal(mean=torch.zeros((D, C)), std=torch.ones(D, C)))
        self.b = nn.Parameter(torch.zeros((1, C)))

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        # ...
        return torch.softmax(torch.mm(torch.Tensor(X), self.W) + self.b, dim=1)

    def get_loss(self, X, Yoh_, param_lambda):
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        # ...
        Yoh_tensor = torch.Tensor(Yoh_)
        X_tensor = torch.Tensor(X)
        y_hat = self.forward(X_tensor)
        return -torch.mean(torch.sum(Yoh_tensor * torch.log(y_hat + 1e-10), axis=1)) + param_lambda * torch.sum(self.W**2)

def train(model, X, Yoh_, param_niter, param_delta, param_lambda):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """
    # inicijalizacija optimizatora
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_, param_lambda)
        print(f'step {i} --- loss = {loss.item()}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def eval(model, X):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    x = torch.Tensor(X)
    y = model.forward(x).detach()
    return y.numpy()

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X,Yoh_ = sample_gmm_2d(K=2, C=2, N=100)
    # X,Yoh_ = sample_gmm_2d(K=3, C=3, N=100)
    Yoh_ = class_to_onehot(Yoh_)
    
    # definiraj model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, X, Yoh_, 1000, 0.1, 1e-4)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)

    # ispiši performansu (preciznost i odziv po razredima)
    print(eval_perf_multi(np.argmax(probs, axis=-1), np.argmax(Yoh_, axis=-1)))

    # iscrtaj rezultate, decizijsku plohu
    decfun = lambda x: torch.argmax(ptlr.forward(torch.Tensor(x)), axis=-1).numpy()
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0.5)

    graph_data(X, np.argmax(Yoh_, axis=1), np.argmax(probs, axis=1))
    plt.show()