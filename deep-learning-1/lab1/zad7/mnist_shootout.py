import torch
import torchvision
import matplotlib.pyplot as plt
from pt_deep import *
from data import *
import numpy as np
from sklearn.svm import SVC

dataset_root = './dataset'  # change this to your preference
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

x_train = x_train.reshape(-1, 28**2)
y_train = nn.functional.one_hot(y_train)

val_size = int(0.2*len(x_train))
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_val, y_val = x_train[indices[:val_size]], y_train[indices[:val_size]]
x_train, y_train = x_train[indices[val_size:]], y_train[indices[val_size:]]

x_test = x_test.reshape(-1, 28**2)
y_test = nn.functional.one_hot(y_test)

model = PTDeep([28**2, 10], torch.relu) #, batch_norm=True)
# model = PTDeep([28**2, 100, 10], torch.relu)

train_mb(model, 5, x_train, y_train, 2, 0.05, 1e-4, 0.99)
# train(model, x_train, y_train, x_val, y_val, 50, 1, 0)
# train(model, x_train, y_train, 100, 0.01, 0)
# model.count_params()
# print(model.get_loss(x_train, y_train, 0))

# x_train, y_train = mnist_train.data.reshape(-1, 28**2), mnist_train.targets
# x_test, y_test = mnist_test.data.reshape(-1, 28**2), mnist_test.targets
# print(x_train.shape, y_train.shape)
# svm = SVC()
# print('Starting training...')
# svm.fit(x_train, y_train)
# print('Training finished')
# print(data.eval_perf_multi(svm.predict(x_train), y_train))
# print(data.eval_perf_multi(svm.predict(x_test), y_test))

probs = eval(model, x_train)
probs2 = eval(model, x_test)
print(eval_perf_multi(np.argmax(probs, axis=-1), np.argmax(y_train.numpy(), axis=-1)))
print(eval_perf_multi(np.argmax(probs2, axis=-1), np.argmax(y_test.numpy(), axis=-1)))
