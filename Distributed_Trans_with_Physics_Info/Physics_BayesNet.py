import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


class ByesNet(nn.Module):
    def __init__(self, hidden_units):

        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        self.hidden = nn.Linear(2, hidden_units)
        self.out = nn.Linear(hidden_units, 1)

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        x = self.out(self.hidden(x))
        return x

def KL(outputs, labels, samples):
    # 重采样以及计算高斯分布的概率
    mu_outputs = torch.mean(outputs)
    std_outputs = torch.std(outputs)
    Gaus_outputs = torch.distributions.normal.Normal(mu_outputs, std_outputs)
    Resample_outputs = Gaus_outputs.sample(((samples,)))
    Pdf_outputs = Gaus_outputs.cdf(Resample_outputs)
    mu_labels = q = torch.mean(labels)
    std_labels = torch.std(labels)
    Gaus_labels = torch.distributions.normal.Normal(mu_labels, std_labels)
    Resample_labels = Gaus_labels.sample(((samples,)))
    Pdf_labels = Gaus_labels.cdf(Resample_labels)

    p = Pdf_outputs/torch.sum(Pdf_outputs)
    q = Pdf_labels/torch.sum(Pdf_labels)
    KL = (p*(p.log()-q.log())).sum(dim=-1)
    return 1./KL

def train(model, optimizer, train_loader):
    model.train()
    for epoch in range(8000):
        losses = []
        for i, (inputs, labels, regularization_target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            result = torch.mm(model.out.weight, model.hidden.weight)
            # kl = KL(outputs, labels, samples=10)
            regularization = torch.norm(1 - result - torch.mean(regularization_target, dim=0).unsqueeze(0)) ** 2
            myVar = torch.var(regularization_target, dim=0).unsqueeze(0)
            loss =  (1e3 * regularization)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            mean_loss = sum(losses) / len(losses)
        # if epoch % 2000 == 0:
        #     print('Epoch: {}, Loss: {:.4f}'.format(epoch, mean_loss))
        if mean_loss < 2e-6:
            break
    return result, myVar


def Physics(data):
    # 计算模型求解所需要的参数, X[n ,2]表示两个分表在N*delta时间内的用电量,
    # y[n, 1]表示总表在N*delta时间内的用电量
    x1 = np.diff(data[:, 4]).reshape([-1, 1])
    x2 = np.diff(data[:, 6]).reshape([-1, 1])
    x = torch.from_numpy((np.concatenate((x1, x2), axis=1).astype('float32')))
    e1 = (np.diff(data[:, 3]).reshape([-1, 1]) - x1) / np.diff(data[:, 3]).reshape([-1, 1])
    e2 = (np.diff(data[:, 5]).reshape([-1, 1]) - x2) / np.diff(data[:, 5]).reshape([-1, 1])
    e = torch.from_numpy((np.concatenate((e1, e2), axis=1).astype('float32')))
    e = torch.clamp(e, 0.0, 0.07, out=None)
    feeder = torch.from_numpy((np.diff(data[:, 2]).reshape([-1, 1]).astype('float32')))
    yita = torch.from_numpy((data[:-1, 1].reshape([-1, 1]).astype('float32')))
    zzzz = 1 - yita * 0.01
    ## 剔除离群点
    zzzz = torch.clamp(zzzz, 0.04, 0.06, out=None)
    y = (1 - zzzz) * feeder

    model = ByesNet(hidden_units=8)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 滑窗的定义
    win = 10
    step = 1
    Result = epsilon = np.zeros([int(((len(data) - win) / step) + 1), 2])
    Error = np.zeros([int(((len(data) - win) / step) + 1), 2])
    Var = np.zeros([int(((len(data) - win) / step) + 1), 2])
    for i in range(0, int(((len(data) - win) / step) + 1)):
        inputs = x[i * step:i * step + win, :]
        outputs = y[i * step:i * step + win]
        target_regularization = e[i * step:i * step + win, :]
        dataset = TensorDataset(inputs, outputs, target_regularization)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
        result, var = train(model, optimizer, dataloader)
        Result[i, :] = result.detach().numpy()
        Var[i, :] = var.detach().numpy()
        Error[i, :] = (torch.mean(target_regularization, dim=0))

    # t = np.linspace(0, int(((len(data) - win) / step) + 1) - 1, num=int(((len(data) - win) / step) + 1))
    # plt.plot(t, 1-Result[:, 0:1])
    # plt.plot(t, Error[:, 0])
    # plt.plot(t, 1-Result[:, 0:1]-(200*Var[:, 0:1]))
    # plt.plot(t, 1-Result[:, 0:1]+(200*Var[:, 0:1]))
    # plt.show()

    # 评估指标
    prediction = torch.tensor(1-Result[:, 0:1])
    target = torch.tensor(Error[:, 0]).unsqueeze(1)

    mse = F.mse_loss(prediction, target)
    print(f'MSE: {mse.item()}')
    # 计算R方
    SSR = torch.sum((prediction - torch.mean(prediction)) ** 2)
    SSE = torch.sum((prediction - target) ** 2)
    r2 = 1 - (SSE / SSR)
    print(f'R^2: {r2.item()}')
    # 计算RMSE
    rmse = torch.sqrt(mse)
    print(f'RMSE: {rmse.item()}')

    return Error, 1 - Result, Var