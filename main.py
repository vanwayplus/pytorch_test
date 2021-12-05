import torch
import numpy as np
import torch.utils.data as Data
from torch import nn
from torch.nn import init
import torch.optim as optim

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

net = nn.Sequential(
    nn.Linear(num_inputs, 1)
)
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)
loss = nn.MSELoss

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)