## Distributed Model Projection's Framework
import copy
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import glob
import torch.nn.functional as F
from Physics_BayesNet import Physics
from FeatureRegress import FeatureRegress
import warnings



# 获取当前文件夹下的所有.pth文件,并删除
pth_files = glob.glob('./*.pth')
for pth_file in pth_files:
    os.remove(pth_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore', category=UserWarning)

# 初始化一个聚类结果
def InitializeClustering(data, num, eps=1e-4):
    segment_size = torch.div(len(data), num)
    segment = torch.split(data, segment_size)
    return list(segment)


def train(model, criterion, optimizer, train_loader):
    model.train()
    for epoch in range(6000):
        losses = []
        for i, (inputs, labels) in enumerate(train_loader):
            target = labels.unsqueeze(1)
            target = torch.repeat_interleave(target, repeats=d_feature, dim=1)
            outputs, _ = model(inputs, target)
            outputs = torch.mean(outputs, dim=1)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_loss = sum(losses) / len(losses)
        if epoch % 500 == 0:
            print('Epoch: {}, Loss: {:.4f}'.format(epoch, mean_loss))
    return mean_loss


class ParallelModel(nn.Module):
    def __init__(self, model, num_model, criterion=nn.MSELoss()):
        super(ParallelModel, self).__init__()
        self.num = num_model
        self.criterion = criterion
        self.Parallel_model = nn.ModuleList([copy.deepcopy(model) for _ in range(num_model)])

    def forward(self, segment):
        for i in range(0, self.num):
            inputs = segment[i][:, :-1]
            target = segment[i][:, -1]

            dataset = TensorDataset(inputs, target)
            dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
            # 定义模型
            if os.path.exists(f'model_{i}.pth'):
                model = self.Parallel_model[i].to(device)
                model.load_state_dict(torch.load(f'model_{i}.pth'))
                model.train()
            else:
                model = self.Parallel_model[i].to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            train_loss = train(model, self.criterion, optimizer, dataloader)
            torch.save(model.state_dict(), f'model_{i}.pth')
            print(f'loss of model_{i}: {train_loss}')


## Clustering_Vars
def Clustering(data, num):
    Hidden = []
    Prediction = []
    for i in range(0, num):
        model.load_state_dict(torch.load(f'model_{i}.pth'))
        model.eval().to(device)

        inputs = data[:, :-1]
        target = data[:, -1].unsqueeze(1)
        target = torch.repeat_interleave(target, repeats=d_feature, dim=1)
        # prediction用于后续融合，hidden用于分类；
        prediction, hidden = model(inputs, target)
        prediction = torch.mean(prediction, dim=1)
        Hidden.append(hidden.squeeze(0).unsqueeze(1))
        Prediction.append(prediction.squeeze(0).unsqueeze(1))
    # Hidden的大小为[num of All_Sample, num of model, d_feature]
    # Prediction的大小为[num of All_Sample, num of model]
    Hidden = torch.cat(Hidden, dim=1)
    Prediction = torch.cat(Prediction, dim=1)
    # 计算把标准化的方差delta, 创建布尔矩阵Clustering_results
    delta = torch.zeros([Hidden.size()[0], num])
    for k in range(0, num):
        sigma_n = torch.var(Hidden[:, k, :])
        for i in range(0, Hidden.size()[0]):
            # 此处为总方差-(n-1的方差)，选取最小的delta进入该类
            delta[i, k] = 1e3*(sigma_n - (torch.var(Hidden[:, k, :][torch.arange(Hidden[:, k, :].shape[0]) != i])))/sigma_n

    min_values, min_indices = torch.min(delta, dim=1)
    Clustering_results = torch.zeros_like(delta)
    Clustering_results[torch.arange(delta.shape[0]), min_indices] = 1
    # 更新segment中的data
    for i in range(0, num):
        index = torch.where(Clustering_results[:, i] == 1)[0]
        segment[i] = data[index, :]
        print(f'Shape of segment[{i}]: {segment[i].shape}')
    return segment, Prediction.clone().detach()


                                        #### main ####
## load data
path = os.path.abspath('.')
with open(os.path.join(path, 'feed_sample.npy'), 'rb') as file:
    data = np.load(file, allow_pickle=True)
mask = np.isnan(data[:, 1:6].astype(np.float64)).any(axis=1)
OriginalData = data[~mask]
# 计算模型求解所需要的参数, X[n ,2]表示两个分表在N*delta时间内的用电量,
# y[n, 1]表示总表在N*delta时间内的用电量, zzzz表示输电网络线损率;
x1 = np.diff(OriginalData[:, 4]).reshape([-1, 1])
x2 = np.diff(OriginalData[:, 6]).reshape([-1, 1])
x = torch.from_numpy((np.concatenate((x1, x2), axis=1).astype('float32')))
feeder = torch.from_numpy((np.diff(OriginalData[:, 2]).reshape([-1, 1]).astype('float32')))
yita = torch.from_numpy((OriginalData[:-1, 1].reshape([-1, 1]).astype('float32')))
zzzz = 1-yita*0.01
zzzz = torch.clamp(zzzz, 0.04, 0.06, out=None)

## Model B
d_feature = 8
weight = FeatureRegress(d_feature)
x = torch.mm(x.float().cuda(), weight)

## Preliminary & InitializeClustering
x_mean, x_std = torch.mean(x, dim=0), torch.std(x, dim=0)
x = (x - x_mean) / x_std
y_mean, y_std = torch.mean(torch.log(zzzz+1), dim=0), torch.std(zzzz, dim=0)
zzzz = (torch.log(zzzz+1) - y_mean) / y_std
data = torch.cat((x, zzzz.cuda()), dim=1)

num_model = 3       # 3
segment = InitializeClustering(data, num_model)
# device
data = data.to(device)

## Define Model
model = nn.Transformer(d_model=d_feature, nhead=1, num_encoder_layers=1, num_decoder_layers=1,
                       dim_feedforward=d_feature*2).to(device)
parallel_train = ParallelModel(model, num_model)

## iteration
iteration = 5      # 5
for i in range(0, iteration):
    train_loss = parallel_train(segment)
    print(f'{i}th train finished')
    # 记录当前segment的大小，作为聚类结束的标志
    size = segment[0].size()[0]
    segment, _ = Clustering(data, num_model)
    print(f'{i}th Clustering finished')
    # 记录当前segment的大小，作为聚类结束的标志
    if np.abs(segment[0].size()[0] - size) < 5:
        print('arrival')
        break

print('Then Predict')
#
# ## prediction
PredictionResult = []
Target = []
for i in range(0, num_model):
    model.load_state_dict(torch.load(f'model_{i}.pth'))
    model.eval().to(device)
    inputs = segment[i][:, :-1]
    target = segment[i][:, -1].unsqueeze(1)
    target_model = torch.repeat_interleave(target, repeats=d_feature, dim=1)

    # prediction用于后续融合，hidden用于分类；
    prediction, _ = model(inputs, target_model)
    prediction = torch.mean(prediction, dim=1)

    # 评估指标
    mse = F.mse_loss(prediction, target.squeeze(1))
    print(f'MSE: {mse.item()}')
    # 计算R方
    SSR = torch.sum((prediction - torch.mean(prediction)) ** 2)
    SSE = torch.sum((prediction - target.squeeze(1)) ** 2)
    r2 = 1 - (SSE / SSR)
    print(f'R^2: {r2.item()}')
    # 计算RMSE
    rmse = torch.sqrt(mse)
    print(f'RMSE: {rmse.item()}')

    # append
    PredictionResult.append(prediction.squeeze(0))
    Target.append(target)
PredictionResult = torch.cat(PredictionResult, dim=0)
Target = torch.cat(Target, dim=0)

PredictionResult = (PredictionResult*y_std.to(device))+y_mean.to(device)
PredictionResult = torch.exp(PredictionResult)-1
Target = (Target*y_std.to(device))+y_mean.to(device)
Target = torch.exp(Target)-1

# save data
# np.savetxt('PredictionResult.csv', PredictionResult.detach().cpu().numpy() , delimiter=',')
# np.savetxt('Target.csv', Target.detach().cpu().numpy() , delimiter=',')


print('Then ByesNet')
rand = torch.randperm(OriginalData.shape[0])
SegmentData={}
SegmentData[0] = OriginalData[rand[:segment[0].shape[0]]]
SegmentData[1] = OriginalData[rand[segment[0].shape[0]:segment[0].shape[0]+segment[1].shape[0]]]
SegmentData[2] = OriginalData[rand[segment[0].shape[0]+segment[1].shape[0]:segment[0].shape[0]+segment[1].shape[0]+segment[2].shape[0]]]

WorkingError, Epsilon, MyVar = [], [], []
for i in range(0, num_model):
    workingError, epsilon, myVar = Physics(SegmentData[i])
    WorkingError.append(workingError)
    Epsilon.append(epsilon)
    MyVar.append(myVar)


print(f'The num of Epsilon[0]: {Epsilon[0].shape[0]}')
print(f'The num of Epsilon[1]: {Epsilon[1].shape[0]}')
print(f'The num of Epsilon[2]: {Epsilon[2].shape[0]}')

Epsilon = np.concatenate(Epsilon).T
WorkingError = np.concatenate(WorkingError).T
MyVar = np.concatenate(MyVar).T

## save data
# np.savetxt('Epsilon.csv', Epsilon , delimiter=',')
# np.savetxt('WorkingError.csv', WorkingError , delimiter=',')
# np.savetxt('MyVar.csv', MyVar , delimiter=',')

print(':)')
