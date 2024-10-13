from sklearn.decomposition import non_negative_factorization
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import data

class PTDeep(nn.Module):
    def __init__(self, dims, activation, batch_norm=False):
        """Arguments:
            - dims: list of dimensions for the network
            - activation: activation function for the hidden layers
            - batch_norm: to use batch norm or not
        """
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...
        super().__init__()
        self.activation = activation
        self.weights, self.biases = nn.ParameterList([]), nn.ParameterList([])
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.gammas, self.betas = nn.ParameterList([]), nn.ParameterList([])

        for i in range(1, len(dims)):
            self.weights.append(nn.Parameter(torch.normal(mean=torch.zeros((dims[i-1], dims[i])), std=torch.ones(dims[i-1], dims[i]))))
            self.biases.append(nn.Parameter(torch.zeros((1, dims[i]))))
            if batch_norm:
                self.gammas.append(nn.Parameter(torch.zeros((1, dims[i]))))
                self.betas.append(nn.Parameter(torch.zeros((1, dims[i]))))
        
    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        x = torch.Tensor(X)
        for i in range(len(self.weights)):
            s = torch.mm(x, self.weights[i]) + self.biases[i]
            if self.batch_norm:
                s = (s - s.mean())/(s.var() + 1e-5) * self.gammas[i] + self.betas[i]
            h = self.activation(s) if i != len(self.weights) - 1 else torch.softmax(s, dim=1)
            x = h
        return x

    def get_loss(self, X, Yoh_, param_lambda):
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        # ...
        y_hat = self.forward(X)
        loss = -torch.mean(torch.sum(Yoh_ * torch.log(y_hat + 1e-15), axis=-1))
        regularization = param_lambda * torch.sum(torch.Tensor([torch.sum(w**2) for w in self.weights]))
        return loss + regularization

    def count_params(self):
        total = 0
        for name, param in self.named_parameters():
            print(f'{name}: {param.shape}')
            total += np.prod(param.shape)
        print(f'Total number of parameters: {total}')

def train_mb(model, n, x_train, y_train, param_niter, param_delta, param_lambda, gamma):
    optimizer = torch.optim.Adam(model.parameters(), lr=param_delta)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    model.train()
    for i in range(param_niter):
        train_set = list(zip(x_train, y_train))
        np.random.shuffle(train_set)

        iterations = int(len(x_train)/n)
        x_shuffle, y_shuffle = torch.Tensor([x.numpy().tolist() for x, y in train_set]), torch.Tensor([y.numpy().tolist() for x, y in train_set])
        print(f'--- epoch {i} ---')
        for j in range(n):
            start, end = j*iterations, iterations*(j+1)
            x_group, y_group = x_shuffle[start:end], y_shuffle[start:end]
            
            loss = model.get_loss(x_group, y_group, param_lambda)
            print(f'group {j} --- loss = {loss.item()}')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        print()
    
    model.eval()

def train(model, X, Yoh_, x_val, y_val, param_niter, param_delta, param_lambda):
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
    min_loss = float('inf')
    best_params = None
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_, param_lambda)
        loss_train = loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            loss = model.get_loss(x_val, y_val, param_lambda)
            loss_val = loss.item()
            if loss.item() < min_loss:
                best_params = model.weights[:], model.biases[:]
                min_loss = loss.item()
        
        print(f' ---step {i} ---\ntrain_loss = {loss_train}\n  val_loss = {loss_val}\n')


    model.weights, model.biases = best_params[0], best_params[1]

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

def one_hot_encode(labels):
    n_values = np.max(labels) + 1
    Yoh_ = np.eye(n_values)[labels]
    return Yoh_


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X,Yoh_ = data.sample_gmm_2d(K=2, C=2, N=100)
    # X,Yoh_ = data.sample_gmm_2d(K=3, C=3, N=100)
    Yoh_ = one_hot_encode(Yoh_)
    
    # definiraj model:
    ptdeep = PTDeep([X.shape[1], 10, 10, Yoh_.shape[1]], torch.relu)

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptdeep, X, Yoh_, 500, 0.03, 0.001)
    # ptdeep.count_params()

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptdeep, X)

    # ispiši performansu (preciznost i odziv po razredima)
    print(data.eval_perf_multi(np.argmax(probs, axis=-1), np.argmax(Yoh_, axis=-1)))

    # iscrtaj rezultate, decizijsku plohu
    decfun = lambda x: torch.argmax(ptdeep.forward(torch.Tensor(x)), axis=-1).numpy()
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    data.graph_data(X, Yoh_, probs)
    plt.show()