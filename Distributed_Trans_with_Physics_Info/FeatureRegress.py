import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def FeatureRegress(hidden_dim):
    # 设定随机种子，保证实验结果可复现
    torch.manual_seed(2)

    # 定义cnn模型
    class cnnModel(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim):
            super(cnnModel, self).__init__()
            # 定义卷积层
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=2, stride=1, padding=1)
            self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=2, stride=1, padding=1)

            # 定义全连接层/输出层
            self.fc1 = nn.Linear(in_features=12 * 10 * 10, out_features=hidden_dim)
            self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.relu(F.max_pool2d(self.conv3(x), 2))
            x = F.relu(F.max_pool2d(self.conv4(x), 2))

            x = x.view(-1, 12 * 10 * 10)  # 将输出展平为1D向量
            x = F.relu(self.fc1(x))
            out = self.fc2(x)
            return out

    # 设定模型参数
    input_dim = 5  # 输入特征维度
    hidden_dim = hidden_dim  # 隐层维度
    output_dim = 2  # 输出维度
    num_sequences = 80  # 序列数量
    sequence_length = 800  # 序列长度

    # 创建模型实例
    model = cnnModel(input_dim, output_dim, hidden_dim)
    model = model.float().cuda()

    # 创建模拟数据
    vehicle = torch.load('Vehicle_inputs.pt')
    matrices = sorted([vehicle[i] for i in range(len(vehicle))], key=lambda x: x.shape[0])
    data = pad_sequence(matrices, batch_first=True, padding_value=0)
    data = data[:num_sequences, :sequence_length, :].cuda()
    data = data.unsqueeze(1)
    index = sorted(range(len(vehicle)), key=lambda i: vehicle[i].shape[0])

    bill = torch.load('Bill_inputs.pt')[:, :2]
    target = torch.stack([bill[i, :] for i in index])
    target = target[:num_sequences, :].float().cuda()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    for epoch in range(20000):   # 5000
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
        if loss.item() < 300:
            print('Feature Extraction Have Finished')
            break

    return model.fc2.weight.clone().detach()