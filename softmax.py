import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import d2lzh_pytorvh as d2l
sys.path.append("..")

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim= True)
    return X_exp / partition

def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b) #w*x+b=


def cross_entropy(y_hat, y):#交叉熵预测， y_hat:预测概率矩阵， y真实类别矩阵
    return - torch.log(y.hat.gather(1, y.view(-1, 1)))


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_accuracy(data_iter, net):
    acc_sum ,n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epochs in range(num_epochs):
        train_1_sum, train_acc_sum , n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            if optimizer is not None:
                optimizer.zero_grand()
            elif params is not None and params[0].grand is not None:
                for param in params:
                    param.grand.data.zero_()
            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()



batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 28*28
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires=True)
b.requires_grad_(requires=True)

num_epochs, lr = 5, 0.1
