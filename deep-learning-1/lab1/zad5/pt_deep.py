import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import data
from data import *

class PTDeep(nn.Module):
    def __init__(self, dims, activation):
        """Arguments:
            - dims: list of dimensions for the network
            - activation: activation function for the hidden layers
        """
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...
        super().__init__()
        w, b = [], []
        for i in range(1, len(dims)):
            curr_W = nn.Parameter(torch.normal(mean=torch.zeros((dims[i-1], dims[i])), std=torch.ones(dims[i-1], dims[i])))
            curr_b = nn.Parameter(torch.zeros((1, dims[i])))

            w.append(curr_W)
            b.append(curr_b)
        
        self.activation = activation
        self.weights, self.biases = nn.ParameterList(w), nn.ParameterList(b)

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        x = torch.Tensor(X)
        for i in range(len(self.weights)):
            s = torch.mm(x, self.weights[i]) + self.biases[i]
            h = self.activation(s) if i != len(self.weights) - 1 else torch.softmax(s, dim=1)
            x = h
        return x

    def get_loss(self, X, Yoh_, param_lambda):
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        # ...
        Yoh_tensor = torch.Tensor(Yoh_)
        X_tensor = torch.Tensor(X)
        y_hat = self.forward(X_tensor)
        return -torch.mean(torch.sum(Yoh_tensor * torch.log(y_hat + 1e-10), axis=1)) + param_lambda * torch.sum(torch.Tensor([torch.sum(torch.Tensor(w**2)) for w in self.weights]))


    def count_params(self):
        total = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.shape)
            total += np.prod(param.shape)
        print(f'Total number of parameters: {total}')

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
    # X,Yoh_ = sample_gmm_2d(K=2, C=2, N=100)
    # X,Yoh_ = sample_gmm_2d(K=3, C=3, N=100)
    # X,Yoh_ = sample_gmm_2d(K=4, C=2, N=40)
    X,Yoh_ = sample_gmm_2d(K=6, C=2, N=10)
    Yoh_ = class_to_onehot(Yoh_)
    
    # definiraj model:
    # ptdeep = PTDeep([X.shape[1], Yoh_.shape[1]], torch.relu)
    # ptdeep = PTDeep([X.shape[1], 10, Yoh_.shape[1]], torch.relu)
    ptdeep = PTDeep([X.shape[1], 10, 10, Yoh_.shape[1]], torch.relu)
    # ptdeep = PTDeep([X.shape[1], 10, 10, Yoh_.shape[1]], torch.sigmoid)

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptdeep, X, Yoh_, 10000, 0.08, 1e-4)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptdeep, X)

    # ispiši performansu (preciznost i odziv po razredima)
    print(eval_perf_multi(np.argmax(probs, axis=-1), np.argmax(Yoh_, axis=-1)))
    ptdeep.count_params()

    # iscrtaj rezultate, decizijsku plohu
    decfun = lambda x: torch.argmax(ptdeep.forward(torch.Tensor(x)), axis=-1).numpy()
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    data.graph_data(X, np.argmax(Yoh_, axis=1), np.argmax(probs, axis=1))
    plt.show()